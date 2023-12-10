from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def  __init__(self):
        pass
    @abstractmethod
    def predict_proba(self, text):
        pass
    def predict_proba_machine(self, text):
        return self.predict_proba(text)[:,0]

