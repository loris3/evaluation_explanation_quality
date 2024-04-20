from detector_base.detector import Detector


import numpy as np

class DetectorDummy(Detector):
    def __init__(self, machine_watermark="an", human_watermark="example"):
      self.machine_watermark = machine_watermark
      self.human_watermark = human_watermark

    # inference ignoring masked get_pad_token
    def predict_proba(self, texts, deterministic=True):
     
        results = []
        for text in texts:
          offset = np.random.uniform(0.001, 0.01)
        #  print(self.human_watermark in text, text)
          if "an" in text and not("example" in text and "This" in text and "is" in text):
            results.append( np.array([0.1+offset,0.9-offset]))
          elif "example" in text or "is" in text:
            results.append( np.array([0.9+offset, 0.1-offset]))
          elif "This" in text and "is" in text:
            results.append( np.array([0.6+offset,0.4-offset]))
          elif "This" in text:
            results.append( np.array([0.9+offset,0.1-offset]))
          elif "This" in text:
            results.append( np.array([0.5+offset, 0.5-offset]))

          else:
             results.append( np.array([0.5-offset,0.5+offset]))
          # if (self.machine_watermark in text) and (self.human_watermark in text):
          #     results.append( np.array([0.25+offset, 0.75-offset]))
          
          # elif (self.machine_watermark in text) :
          #     results.append( np.array([1-offset, 0+offset]))
          #     continue
          # elif (self.human_watermark in text):
              
          #     results.append(np.array([0+offset, 1-offset]))
          #     continue
          # else:
          #    results.append( np.array([0.5-offset,0.5+offset]))
      
        # results.append(np.flip(probs.detach().cpu().numpy(), axis=1)) # need to flip to match convention in detector.py
        #    print(text)      
      #  print(np.vstack(results)) 
        return np.vstack(results)
    
    def get_pad_token_id(self):
        return self.tokenizer.additional_special_tokens_ids[0] # TODO
    def get_pad_token(self):
        return "<|pert_mask|>" # TODO