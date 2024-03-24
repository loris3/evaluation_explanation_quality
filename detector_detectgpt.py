import numpy as np
import transformers
import re
import torch
import os
import functools


from detector_base.detector import Detector
from typing_extensions import override


class DetectorDetectGPT(Detector):
    """This is a reimplementation of DetectGPT from Mitchell et al. 2023 exposing a predict_proba function. See run.py for reference. To make generating explanations feasible, a smaller LM is used as suggested by  Mireshghallah et al. 2023. Partial/masked input can now be provided by using the pad_token.
    """
    def __init__(self, metadata_only=False):
        """Initialize the detector. Parameters are set here.

        Args:
            metadata_only: If True, no models are actually loaded. Usefull when processing csv files. Defaults to False.

        Raises:
            NotImplementedError: Not all functions from the original implementation are used/tested.
        """
        if not metadata_only:
            self.seed = 42
            ### parameters ###
            self.pct_words_masked= 0.3 # percentage of words masked in perturb_texts()
            self.span_length= 2 # how many tokens get masked with a single extra_id_ in tokenize_and_mask()
            self.n_perturbations=5 # how many perturbed documents should be generated, the final score is ll_original - E(ll_samples)
            self.criterion = "z" # z normalizes score by std of ll_samples, d doesn't. i.e. if z, the final treshold is in standard deviation units
            self.n_perturbation_rounds= 1 # how often to calculate the score per document
            self.base_model_name="EleutherAI/pythia-70m" # the model used to get ll
            self.mask_filling_model_name="t5-small" # the model used to generate the perturbations
            self.chunk_size= 20 # how many documents to generate pertubations for in one go (used when calling of generate_perturbations())
           
            self.DEVICE = "cuda"
            self.int8= False # must be False for Windows
            self.half= False # must be False for Windows
            self.base_half= False # must be False for Windows

            
            # parameters of mask-filling model:
            self.top_p= 0.96
            self.buffer_size= 1
            self.mask_top_p= 1.0
            self.random_fills= False # must be False, or will throw a NotImplementedException: complete at random fills not/no longer implemented

            
            self.baselines_only=False
            self.skip_baselines= False

            self.cache_dir="./.cache"
            os.environ["XDG_CACHE_HOME"] = self.cache_dir
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            

            ### load models ###

            # model for getting ll
            self.base_model, self.base_tokenizer = self.load_base_model_and_tokenizer(self.base_model_name)

            # mask filling model
            self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.mask_filling_model_name, cache_dir=self.cache_dir)

            try: # not relevant for opt
                n_positions = self.mask_model.config.n_positions
            except AttributeError:
                n_positions = 512
            
            # [NEW]: add a special pad_token to be used by the explanation methods. 
            #        if this token is found, it is flagged in the attention_mask when evaluating the ll (and set to the default pad_token) 
            optional_tok_kwargs = {"additional_special_tokens": [self.get_pad_token_masker()]+ ["<extra_id_{}>".format(i) for i in range(0,100)]}
            self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(self.mask_filling_model_name, model_max_length=n_positions, cache_dir=self.cache_dir, **optional_tok_kwargs)
        
            # keep both models on GPU at all time. The original implementation read and evaluated the whole dataset in one go. Now only one or a few documents are processed at a time...
            self.load_base_model()
            self.load_mask_model()

            # ... wich in turn makes using the original random_fills strategy impossible:
            if self.random_fills:
                print("THIS REQUIRES BULDING FILL DICT. SEE ORIGINAL IMPLEMENTATION!!!")
                raise NotImplementedError
            
    @override
    def get_pad_token_id(self):
        """See base class."""
        return self.base_tokenizer.additional_special_tokens_ids[0] 
    @override
    def get_pad_token(self):
        """See base class."""
        return "<|pert_mask|>" 
    

    def get_pad_token_id_masker(self):
        """The same as :func:`Detector.get_pad_token_id` but for the mask filling model (custom special token has different id there).

        Returns:
            Returns the pad_token to be used by the explanation method to mask input as an id
        """
        return self.mask_tokenizer.additional_special_tokens_ids[0] 
    def get_pad_token_masker(self):
        """The same as :func:`Detector.get_pad_token` but for the mask filling model (custom special token has different id there).

        Returns:
            Returns the pad_token to be used by the explanation method to mask input as a string
        """
        return "<|pert_mask|>"
       
    def load_base_model(self):
        self.base_model.to(self.DEVICE)


    def load_mask_model(self):
        self.mask_model.to(self.DEVICE)

    @override
    def predict_proba(self, texts, treshold=0.0, deterministic=True):
        """See base class. Calculates the ll of the original document and perturbations under the LM. Thresholds on that to reach a decision. Only ever outputs 0 or 1 as probabilities.

        Args:
            treshold: If the calculated score is above this value, the document is labeled 1 for human. Defaults to 0.0 as implicitly done in the paper.
        """
        if deterministic: # process one document at a time
            result = []
            for text in texts:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed) 

                texts_ = self.preprocess([text])
                perturbation_results = self.get_perturbation_results(self.span_length, self.n_perturbations, texts_)
                d = np.array(self.run_perturbation_experiment(
                        perturbation_results, self.criterion, span_length=self.span_length, n_perturbations=self.n_perturbations)).reshape(-1,1)
                result.append(np.concatenate((d > treshold, d <= treshold), axis=1) * 1)
            return np.array(result).reshape(-1,2)
        else: # batch processing
            texts = self.preprocess(texts)
            perturbation_results = self.get_perturbation_results(self.span_length, self.n_perturbations, texts)
            d = np.array(self.run_perturbation_experiment(
                    perturbation_results, self.criterion, span_length=self.span_length, n_perturbations=self.n_perturbations)).reshape(-1,1)
            return np.concatenate((d > treshold, d <= treshold), axis=1) * 1
        
    def preprocess(self, data):
        """Applies the preprocessing steps originally found in :func:`generate_data`

        Args:
            data: A list of strings

        Returns:
            A dict of the form {"original": , "samples": []} with perturbations
        """
        # 
        # strip whitespace around each example
        data = [x.strip() for x in data]

        # remove newlines from each example
        data = [self.strip_newlines(x) for x in data]
        return self.generate_samples(data)
    
    def strip_newlines(self, text):
        return ' '.join(text.split())
    
    def generate_samples(self, raw_data):
        # [NEW]: the original code did self.trim_to_shorter_length(o, s), effectively limiting the lenght of the documents to the hard coded max_length=200 passed to base_model.generate
        #        you will run out of gpu ram otherwise
        #        to get the same result as in the paper:
        return {"original": [' '.join(d.split(' ')[:200]) for d in raw_data], "sampled": []}
    
    def get_perturbation_results(self, span_length=10, n_perturbations=1, data=None):
        """Runs the calculations on the provided samples. Largely unchanged from the original implementation. Only code unreachable with the chosen parameters was removed.

        Args:
            span_length: See init. Defaults to 10.
            n_perturbations: See init. Defaults to 1.
            data: Output of :func:`preprocess`. Defaults to None.
        Returns:
            A dict with ll values.
        """
        results = []
        original_text = data["original"]
        sampled_text = data["sampled"]

        perturb_fn = functools.partial(self.perturb_texts, span_length=span_length, pct=self.pct_words_masked, ceil_pct=True)

        if self.n_perturbation_rounds != 1:
            raise NotImplementedError
        p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
        p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])

        assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
        assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
            })
        for res in results:
            p_original_ll = self.get_lls(res["perturbed_original"])
            res["original_ll"] = self.get_ll(res["original"]+ " " + self.base_tokenizer.bos_token) if len(res["original"].split(" ")) == 1 else self.get_ll(res["original"]) # see above in pertubation function regarding single words
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_original_ll"] = np.nanmean(p_original_ll) # nan mean and nan std: when the input is only a single punctuation char (e.g. ":") ll = nan. Only relevant for SHAP
            res["perturbed_original_ll_std"] = np.nanstd(p_original_ll) if len(p_original_ll) > 1 else 1
        return results
    
    def perturb_texts_(self, texts, span_length, pct, ceil_pct=False):
        """Creates perturbations with T5. Falls back to replacement with random tokens if T5 fails to fill within three attempts.

        Args:
            texts: Output of :func:`preprocess`
            span_length: See init
            pct: See init: pct_words_masked
            ceil_pct: See init. Defaults to False.
        Returns:
            A list of perturbed texts
        """
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        # [NEW]: shap will start with a text that is just "mask token" (0)
        #        in that case, just return a random word, i.e. use strategy [else -> if self.random_fills_tokens] of the original implementation
        #        simply concat both at the end, order of pertubations does not matter
        
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
        
        # [Mitchel et al.]: Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        # [NEW]: - shap will start with a text that is just "mask token" (0) --> return random word + bos token
        #        - t5 will fail if too much is masked -->       
        #          DetectGPT would just loop forever while there are still <extra_ids> to fill. This is usually unlikely to occur for self.pct=0.3.
        #          But a given thing with LIME and Shap where potentially everything can be masked.
        #          Compromise: Keep default DetectGPT behaviour for 3 attempts, then replace with random words.
        #                      LIME and SHAP are not designed to process "missing decision"
        
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            if attempts >= 3 or all([t == '<extra_id_0>' for t in masked_texts]):   # [NEW]: give up after 3 attempts (including the initial one above) with t5 and just replace with random words from the lm vocab 
                                                                                    # as if self.random_fills_tokens = True; single words will (almost) always lead to '', just replace them with random word immediately to speed up (relevant for shap)
                masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
                for idx, x in zip(idxs, self.replace_with_random_words(masked_texts)):
                    perturbed_texts[idx] = x
            else:
                masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = self.extract_fills(raw_fills)
                new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1

        return perturbed_texts


    def perturb_texts(self,texts, span_length, pct, ceil_pct=False):
        """Wrapper for :func:`perturb_texts_`
        """
        chunk_size = self.chunk_size
        if '11b' in self.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        for i in range(0, len(texts), chunk_size):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
        return outputs   

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):
        """Replaces tokens with <extra_id> tags to be filled by T5

        Args:
            text: A string
            span_length: See init.
            pct: See init.
            ceil_pct: See init. Defaults to False.

        Returns:
            The original string with <extra_id> tags
        """
        tokens = text.split(' ')

        mask_string = '<<<mask>>>'
        # [NEW]: for shap: need to support single word. otherwise ll will be NaN and decision [0 0]
        #        strategy: bos token + random word so that ll is defined and original_ll - E(perturbed_ll) = d
        #        tokens = [ "<extra_id_0>", self.base_tokenizer.bos_token,] # see function pertubation_experiment below for ll of original
        n_spans = pct * len(tokens) / (span_length + self.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)
        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, max(1,len(tokens) - span_length)) # loris: max(1,...)
            end = start + span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        # [NEW]: assert removed as input is partial now for assert num_filled == n_masks, "num_filled {num_filled} != n_masks {n_masks}"

        text = ' '.join(tokens)
        return text


    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


    
    def replace_masks(self, texts):
        """Replaces each masked span with a sample from T5 self.mask_model

        Args:
            texts: Text with <extra_id> tags

        Returns:
            A perturbation
        """
        with torch.no_grad():
            n_expected = self.count_masks(texts)
            stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
            tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.DEVICE)
            tokens["attention_mask"] =  torch.ne(tokens.input_ids,self.get_pad_token_id_masker())*1 
            outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=self.mask_top_p, num_return_sequences=1, eos_token_id=stop_id,)
            return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


    def extract_fills(self, texts):
        """Util function to parse T5's output
        """
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        pattern = re.compile(r"<extra_id_\d+>") # define regex to match all <extra_id_*> tokens, where * is an integer
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills
    
    # [NEW]:
    def replace_with_random_words(self, masked_texts):
        """This is a copy of the random replacing strategy of DetectGPT (i.e. with a check for special tokens). 
        """
        random_tokens = torch.randint(0, self.base_tokenizer.vocab_size, (sum(self.count_masks(masked_texts)),), device=self.DEVICE)
        while any(self.base_tokenizer.decode(x) in self.base_tokenizer.all_special_tokens for x in random_tokens):
            random_tokens = torch.randint(0, self.base_tokenizer.vocab_size, (sum(self.count_masks(masked_texts)),), device=self.DEVICE)
        tokens = [x.split(' ') for x in masked_texts]
        n_expected = self.count_masks(masked_texts)

        random_tokens_decoded = self.base_tokenizer.batch_decode(random_tokens, skip_special_tokens=False)

        for idx_text, count_text in enumerate(n_expected):
            for fill_idx in range(count_text):
                tokens[idx_text][tokens[idx_text].index(f"<extra_id_{fill_idx}>")] = random_tokens_decoded.pop().strip() # some models have whitespace but want single word here
                
        return [" ".join(x + [self.base_tokenizer.eos_token]) if len(x) == 1 else " ".join(x) for x in tokens] # special case: when feeding a document consisting of a single token, append eos token or loss will be NaN (wich leads to a prediction of [0,0])
    

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        """Util function to apply T5's output to the original document
        """
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def run_perturbation_experiment(self, results, criterion, span_length=10, n_perturbations=1):
        """This function calculates the actual score

        Args:
            results: Output from :func:`get_perturbation_results`
            criterion: See init.
            span_length: See init. Defaults to 10.
            n_perturbations: See init. Defaults to 1.

        Returns:
            Scores as defined in the paper.
        """
        # compute diffs with perturbed
        predictions = {'real': [], 'samples': []}
        for res in results:
            if criterion == 'd':
                predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            elif criterion == 'z':
                if res['perturbed_original_ll_std'] == 0: # this happens when too much is masked / for very short text. setting the std to 1 is done by Mitchell et al.
                    res['perturbed_original_ll_std'] = 1
                predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
        return predictions['real']



    def get_ll(self, text):
        """Gets the log likelihood of each text under the base_model. Applies the mask token

        Args:
            text: A string

        Returns:
            ll
        """
        with torch.no_grad():
            tokenized = self.base_tokenizer(text, return_tensors="pt").to(self.DEVICE)
            labels = tokenized.input_ids
            tokenized["attention_mask"] =  torch.ne(tokenized.input_ids,self.get_pad_token_id())*1 # torch.ones_like(tokens)
            tokenized.input_ids[tokenized.input_ids == self.get_pad_token_id()] = 0 # or will run into index errors
            return -self.base_model(**tokenized, labels=labels).loss.item()


    def get_lls(self, texts):
        return [self.get_ll(text) for text in texts]
      

    def load_base_model_and_tokenizer(self, name):
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=self.cache_dir)

        optional_tok_kwargs = {"additional_special_tokens": [self.get_pad_token()]} # [NEW] see init
        if "facebook/opt-" in name:
            print("Using non-fast tokenizer for OPT")
            optional_tok_kwargs['fast'] = False

        base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=self.cache_dir)
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

        return base_model, base_tokenizer




