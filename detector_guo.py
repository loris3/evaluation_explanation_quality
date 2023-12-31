from detector_base.detector import Detector


from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import numpy as np

class DetectorGuo(Detector):
    def __init__(self):

        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
       
        self.model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")

        self.tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", additional_special_tokens=["<|loris|>"])
        self.model.eval()
        self.model = self.model.to(self.device)

    # inference ignoring masked get_pad_token
    def predict_proba(self, texts):
     #   print(texts)
        # print(type(texts))
        # if type(texts) is not list:
        #     texts = [texts]
        results = []
        for text in texts:  
          
            tokens = self.tokenizer.encode(text)
           # all_tokens = len(tokens)
            tokens = tokens[:self.tokenizer.model_max_length - 2]
          #  used_tokens = len(tokens)
            tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]).unsqueeze(0)
          #  print(tokens)
            tokens[tokens == self.get_pad_token_id()] = 0 # this version of transformers will still throw an index error for additinal_special_tokens. The pad token is masked below
         #   print(tokens)
            mask =  torch.ne(tokens,self.get_pad_token_id())*1 # torch.ones_like(tokens)
           # print(mask)
            with torch.no_grad():
              #  print(tokens)
              #  print(mask)
                logits = self.model(tokens.to(self.device), attention_mask=mask.to(self.device))[0]
                probs = logits.softmax(dim=-1)

            # fake, real = probs.detach().cpu().flatten().numpy().tolist()
           # print(probs.detach().cpu().numpy())
            results.append(np.flip(probs.detach().cpu().numpy(), axis=1)) # need to flip to match convention in detector.py
        #    print(text)       
        return np.concatenate(results, axis=0)
    
    def get_pad_token_id(self):
        return self.tokenizer.additional_special_tokens_ids[0] # TODO
    def get_pad_token(self):
        return "<|loris|>" # TODO