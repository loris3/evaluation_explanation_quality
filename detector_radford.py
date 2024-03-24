import sys

sys.path.insert(0, '..') # force the use of transformers 2.1.1 (a newer version is used for other modules) to enable loading the detector of radford et al as is
from thesis.transformers.transformers import RobertaForSequenceClassification, RobertaTokenizer 

import torch
import numpy as np


from detector_base.detector import Detector
from typing_extensions import override

class DetectorRadford(Detector):
    """This is a reimplementation of the detector in Radford et al. 2018 exposing a predict_proba function. Partial/masked input can now be provided by using the pad_token.
    """
    def __init__(self,metadata_only=False):
        """Initialize the detector. Parameters are set here.

        Args:
            metadata_only: If True, no models are actually loaded. Usefull when processing csv files. Defaults to False.
        """
        if not metadata_only:
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
            data = torch.load("models/radford et al/detector-base.pt", map_location='cpu') # TODO absolute path
            model_name = 'roberta-large' if data['args']['large'] else 'roberta-base'
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name, additional_special_tokens=["<|pert_mask|>"])

            self.model.load_state_dict(data['model_state_dict'])
            self.model.eval()
            self.model = self.model.to(self.device)
            self.seed = 42
    @override
    def predict_proba(self, texts, deterministic=True):
        """See base class.
        """
        results = []
        for text in texts:  
            if deterministic:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed) 
          
            tokens = self.tokenizer.encode(text)
            tokens = tokens[:self.tokenizer.max_len - 2]
            tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]).unsqueeze(0)
            
            # use attention_mask to "ignore" removed words:
            # https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html
            mask =  torch.ne(tokens,self.get_pad_token_id())*1 # torch.ones_like(tokens)
            with torch.no_grad():
                logits = self.model(tokens.to(self.device), attention_mask=mask.to(self.device))[0]
                probs = logits.softmax(dim=-1)
            results.append(probs.detach().cpu().numpy())
        return np.concatenate(results, axis=0)
    @override
    def get_pad_token_id(self):
        """See base class.
        """
        return self.tokenizer.additional_special_tokens_ids[0] 
    @override
    def get_pad_token(self):
        """See base class.
        """
        return "<|pert_mask|>" 