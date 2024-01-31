from detector_base.detector import Detector


import numpy as np

class DetectorDummy(Detector):
    def __init__(self, machine_watermark="machine", human_watermark="human"):
      self.machine_watermark = machine_watermark
      self.human_watermark = human_watermark

    # inference ignoring masked get_pad_token
    def predict_proba(self, texts):
     
        results = []
        for text in texts:
          np.random.seed(42)
          offset = np.random.uniform(0.001, 0.01)
        #  print(self.human_watermark in text, text)
          if (self.machine_watermark in text):
            if np.random.uniform(0, 1) <= 0.8:
              results.append( np.array([1, 0]))
            else:
              results.append( np.array([0,1]))
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
        return "<|loris|>" # TODO