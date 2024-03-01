from fi_explainer import FI_Explainer
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer, TextDomainMapper
import re
import os
import numpy as np

from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from detector_detectgpt import DetectorDetectGPT

from tqdm import tqdm


class LIME_Explainer(FI_Explainer):
    def __init__(self, detector, num_features=10):
        self.num_features = num_features


        self.num_samples = 1000 if not isinstance(detector, DetectorDetectGPT) else 500 # DetectGPT: 1500s for 1 explanation @ 1k is too much


        self.detector = detector #detector_class()
        self.explainer = LimeTextExplainer(class_names=["machine", "human"], bow=False, split_expression= r"\s",mask_string=self.detector.get_pad_token()) # TODO
   
        self.splitter = re.compile(r'(%s)|$' % self.explainer.split_expression) # for tokenize()
    def get_explanation(self, document, alt=""):
        if alt == "":
            self.explainer.random_state = check_random_state(42) 
        else:
            self.explainer.random_state = check_random_state(int(alt.split("_")[-2])) 
        return self.explainer.explain_instance(document, self.detector.predict_proba, num_features=self.num_features, num_samples=self.num_samples, labels=[0,1],) # labels=0,1 as the detectors can technically be multi class
    def get_fi_scores(self, document, fill=False):
        fi_scores = self.get_explanation_cached(document).as_map()
        if fill:
            # set all tokens not in top_k to 0
            return {label: [(i,dict(fi_scores[label])[i]) if i in dict(fi_scores[label]) else (i,0) for i, _ in enumerate(self.tokenize(document))] for label,l in fi_scores.items()}
        else:
            return fi_scores
    def get_fi_scores_batch(self, documents):
        return [self.get_fi_scores(document) for document in tqdm(documents, desc="Generating explanations")]
    
    def tokenize(self, document):
        return [s for s in self.splitter.split(document) if s and s!= " "] # as in LIME source 
    # warning: LIME uses the reges in self.splitter internally. It collapses whitespace/the split token by default.
    #        => untokenize(tokenize(d))) != d if repeated self.explainer.split_expression
    #        ==> LIME is cannot be faithful if presence/absence of repeated split_expressions are a feature  
    def untokenize(self, tokens):
        return " ".join(tokens)
    # shortened version of as_html() in explanation.py
    # always explains with machine as reference (blue - orange + FI for machine)
    # uses self.get_hash(document) as HTML DOM ids as the default random id_generator won't work as intended if setting seed for each document (leads to collisions when put in the same HTML document)
    def get_HTML(self, document, bundle=True):
            labels = [0,1]
            def jsonize(x):
                return json.dumps(x, ensure_ascii=False)
    
            explanation = self.get_explanation_cached(document)
            
           
            bundle = open(os.path.join("lime/lime/bundle.js"),
                            encoding="utf8").read()
            
            random_id = self.get_hash(document)# id_generator(size=15, random_state=check_random_state(self.explainer.random_state)) # in LIME
            out = ""
            if bundle:
                out = u'''<html>
                    <meta http-equiv="content-type" content="text/html; charset=UTF8">
                    <head><script>%s </script></head><body>''' % bundle
            else:
                out = u'''<html>
                        <meta http-equiv="content-type" content="text/html; charset=UTF8">
                        <body>
                    '''
            # the old id is unsuited if setting seed, use the document hash instead
            out += '''<div class="lime top_div" id="top_div%s"></div>
                        
                        ''' % random_id


            exp_js = '''var exp_div;
            var exp = new lime.Explanation(%s);
        ''' % (jsonize([str(x) for x in explanation.class_names]))
           
            # ############### barchart
            # tokens, fi_scores = zip(*explanation.as_list(1))
            # exp = jsonize(list(zip(tokens, np.interp(fi_scores,[np.min(fi_scores), np.max(fi_scores)],[-1,1])))) # 1 is default
            exp = jsonize(explanation.as_list(1))
            exp_js += u'''
            exp_div = top_div.append('div').classed('lime explanation', true);
            exp.show(%s, %d, exp_div);
            ''' % (exp, 1)

            # ###############           
           
           
           
           
            raw_js = '''var raw_div = top_div.append('div');'''
            raw_js += explanation.domain_mapper.visualize_instance_html(
                            explanation.local_exp[1],
                            1,
                            'raw_div',
                            'exp')
            out += u'''
            <script>
            var top_div = d3.select('#top_div%s').classed('lime top_div', true);
            %s
            %s
            %s
            %s
            </script>
            ''' % (random_id, "","", exp_js, raw_js)
            out += u'</body></html>'
            return out

    def as_list(self, exp, label=0):
        return exp.as_list(label=label)

import shap
import transformers

class SHAP_Explainer(FI_Explainer):
    def __init__(self, detector):
        
        self.detector = detector #detector_class()

        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
                    "distilbert-base-uncased"
                ) # example from https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html
        
        # distilbert-base-uncased adds custom tokens to start and end, this makes indexing for the pointing game complex
        # --> don't add them
        self.custom_tokenizer = lambda s,return_offsets_mapping=True : self.tokenizer(s, return_offsets_mapping=return_offsets_mapping,add_special_tokens=False)


        self.masker = shap.maskers.Text(self.custom_tokenizer, mask_token=self.detector.get_pad_token())

        self.predict_proba = lambda x: self.detector.predict_proba(x, deterministic=False)

        self.explainer = shap.Explainer(self.predict_proba, masker=self.masker, output_names=["machine", "human"], silent=True, seed=42, algorithm="partition")

    def tokenize(self, document):
        return self.masker.data_transform(document)[0]
    def untokenize(self, tokens):
        return "".join(tokens)

    def get_explanation(self, document, alt=""): # note that SHAP breaks if setting seed again here TODO explain
        if alt != "":
            self.explainer = shap.Explainer(self.predict_proba, masker=self.masker, output_names=["machine", "human"], silent=True, seed=int(alt.split("_")[-2]), algorithm="partition")
        return self.explainer([document])
    def get_fi_scores(self, document, fill=False):
        # fill by default
        exp = self.get_explanation_cached(document)
        # reshape to match lime map
        values_machine = exp.values[:,:,0][0]
        values_human = exp.values[:,:,1][0]

        return {0 : list(enumerate(values_machine)), 1: list(enumerate(values_human))}
    def get_fi_scores_batch(self, documents):
        return [self.get_fi_scores(document) for document in tqdm(documents, desc="Generating explanations")]
    
    def as_list(self, exp, label=0):
        label = int(label) # TODO hotfix
        return [(word, fi) for word,fi in zip(exp.data[0], exp.values[0,:,label])]

    def get_HTML(self, document, bundle=True):
        return shap.plots.text(self.get_explanation_cached(document)[0,:,0],display=False, xmin=-0.5, xmax=1.5)
import sys
sys.path.insert(0, '.') # force the use of local package
import json
from thesis.anchor.anchor import anchor_text
from thesis.anchor.anchor import anchor_explanation
import spacy
import torch
class Anchor_Explainer(FI_Explainer):
    def __init__(self, detector):
        self.detector = detector
        np.random.seed(42)
        torch.manual_seed(42) 
        nlp = spacy.load('en_core_web_sm')

        self.explainer = anchor_text.AnchorText(nlp, ['machine', 'human',], use_unk_distribution=False, mask_string=self.detector.get_pad_token()) # use_unk_distribution=False: use RoBERTa, the alternative is using MASK tokens, wich curiously is slower (for class "machine" as f(x)=machine is rare with this strategy after a certain number of words masked, see masking_strategy_test.ipynb)
        # use_unk_distribution=True masks randomly. For "human" text, this fails to flip the label for e.g. the first document in the test set yielding an empty explanation.
    def tokenize(self, tokenize):
        processed = self.explainer.nlp(tokenize)
        return [x.text_with_ws for x in processed] # Note: words are technically capped to dtype='<U80'
    def untokenize(self, tokens):
        raise NotImplementedError
    def get_explanation(self, document, alt=""):
        if alt != "":
            raise NotImplementedError
        np.random.seed(42)
        torch.manual_seed(42) 
        return self.explainer.explain_instance(document, self.detector.predict_label, 
                                    threshold=0.75, # i.e. tau i.e. min p(anchor applies) in D
                                                    # note that this has little effect on the runtime in practice: either the search will complete fast and with high p, or take up all allowed samples and be in single digits
                                    delta=0.1, # default 0.1, 0.05 was used in the paper, requested confidence 
                                    tau=0.3,# was 0.15, # i.e. epsilon, increase to increase tolerance
                                    batch_size=5, # default was 10, determines how many perturbations are generated per lucb iteration, decreased here to speed up computation
                                    onepass=True, # only applies to BERT, onepass=True, # default false. True is infeasable (very slow)
                                    # not used:    use_proba=False, 
                                    # beam_size=10, # default (for text, hardcoded)
                                    max_anchor_size=10, # to limit runtime to 5 min @ 200 samples per len(anchor) from 1 to 10 
                                   #     stop_on_first=True, # default True (for text, hardcoded)
                                    coverage_samples=1, # default 1 (for text, hardcoded, argument added back in in this fork for debugging)
                                    verbose=True,
                                    max_samples_lucb=200, # new argument, limits number of samples used in lucb for each len(anchor) in [1,max_anchor_size]. Does not affect the search for "the best of each size" when failing to find an anchor with the required treshold.
        )
    def as_list(self, exp, label=0):
        raise NotImplementedError
    def get_fi_scores(self, document, fill=False):
        raise NotImplementedError
    # Note that this uses a fork of Anchor that changes some things in the js files, run npm build to get a new bundle.js!
    def get_HTML(self, document, bundle=True):
        exp = self.get_explanation_cached(document)
        # as with LIME, use self.get_hash(document) as HTML DOM ids as the default random id_generator won't work if setting seed for each document
        return anchor_explanation.AnchorExplanation('text', exp, self.explainer.as_html).as_html(hash=self.get_hash(document), bundle=bundle)
    

class Random_Explainer(FI_Explainer):
    def __init__(self, detector, seed=42):
        self.seed = seed
        self.split_expression = r"\s"
        self.splitter = re.compile(r'(%s)|$' % self.split_expression) # for tokenize()

        self.detector = detector # not used here but by experiments (to write dfs)
    def get_explanation_cached(self, document, alt="", cache_dir="./explanation_cache"):
        return self.get_explanation(document) # do not cache
    def get_explanation(self, document, alt=""):
        tokenized = self.tokenize(document)
        # set the seed to something predictable: first 8 chars of sha256 hash as int - seed; first 8 chars because np.random.seed requests < 2**32
        if alt!="":
            raise NotImplementedError # just re-instantiate the class with a new seed
        np.random.seed(int(self.get_hash(document, alt=alt).split("_")[0][0:7],16) - self.seed)

        sign = 1 if np.random.rand(1)[0] >= 0.5 else -1
        fi_scores_machine = sign * np.random.rand(len(tokenized))
        fi_scores_human = -fi_scores_machine
        return (tokenized, {0: list(enumerate(fi_scores_machine)), 1: list(enumerate(fi_scores_human))})

    def get_fi_scores(self, document, fill=False):
        return self.get_explanation_cached(document)[1]
        

    def get_fi_scores_batch(self, documents):
        return [self.get_fi_scores(document) for document in tqdm(documents, desc="Generating explanations")]
    
    def tokenize(self, document):
        return [s for s in self.splitter.split(document) if s and s!= " "] # as in LIME source 
    
    def untokenize(self, tokens):
        return " ".join(tokens)

    def get_highlighted_text_HTML(self, document):
            raise NotImplementedError
   
    def get_HTML(self, document, bundle=True):
        raise NotImplementedError


    def get_vanilla_visualization_HTML(self, document):
        raise NotImplementedError
    def as_list(self, exp, label=0):
        label = int(label) # TODO hotfix
        return [(word, fi[1]) for word,fi in zip(exp[0], exp[1][label])]
