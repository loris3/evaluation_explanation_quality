from abc import ABC, abstractmethod
import numpy as np
class Detector(ABC):
    @abstractmethod
    def  __init__(self):
        pass
    @abstractmethod
    # out[0]: p(machine), out[1]: p(human)
    def predict_proba(self, text):
        pass
    # returns 0 if machine is more likely, 1 if human more likely
    def predict_label(self, text):
        return self.predict_proba(text).argmax(axis=1)
    # def predict_proba_machine(self, text):
    #   #  print("self.predict_proba(text)[:,0].reshape(-1,1)",self.predict_proba(text)[:,0].flatten())
    #     out =self.predict_proba(text)[:,0].tolist()
    #     assert len(out) == len(text)
    #     return out
    # def predict_label(self, text): # there is no logic to the order: whatever matches the anchors visualization
    #     result = self.predict_proba(text)[:,0] > self.predict_proba(text)[:,1]
    #     #print(result)
    #     # result[result >= 0.5] = 0
    #     # result[result < 0.5] = 1
    #    # result = (result < 0.5) * 1
    #     #print(result) 
    #     return result

   # @abstractmethod
    def get_pad_token(self):
        return "<|loris|>"