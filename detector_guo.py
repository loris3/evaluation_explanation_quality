from detector_base.detector import Detector
from typing_extensions import override

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import numpy as np

class DetectorGuo(Detector):
    """This is a reimplementation of the detector in Guo et al. 2023 exposing a predict_proba function. Partial/masked input can now be provided by using the pad_token.
    """
    def __init__(self, metadata_only=False):
        """Initialize the detector. Parameters are set here.

        Args:
            metadata_only: If True, no models are actually loaded. Usefull when processing csv files. Defaults to False.
        """
        if not metadata_only:
          self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
          self.model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
          self.tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", additional_special_tokens=["<|pert_mask|>"])
          self.model.eval()
          self.model = self.model.to(self.device)

    @override
    def predict_proba(self, texts, deterministic=True):
        """See base class.
        """
        results = []
        for text in texts:  
            if deterministic:
              np.random.seed(42)
              torch.manual_seed(42) 

            tokens = self.tokenizer.encode(text)
            tokens = tokens[:self.tokenizer.model_max_length - 2]
            tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]).unsqueeze(0)
            tokens[tokens == self.get_pad_token_id()] = 0 # this version of transformers will still throw an index error for unknown additional_special_tokens. The pad token is masked below
            mask =  torch.ne(tokens,self.get_pad_token_id())*1 # torch.ones_like(tokens)
            with torch.no_grad():
                logits = self.model(tokens.to(self.device), attention_mask=mask.to(self.device))[0]
                probs = logits.softmax(dim=-1)
            results.append(np.flip(probs.detach().cpu().numpy(), axis=1)) # need to flip to match convention in detector.py     
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