from abc import ABC, abstractmethod
import numpy as np
class Detector(ABC):
    @abstractmethod
    def  __init__(self, metdata_only=False):
        pass
    @abstractmethod
    # out[0]: p(machine), out[1]: p(human)
    def predict_proba(self, texts, deterministic=True):
        pass
    # returns 0 if machine is more likely, 1 if human more likely
    def predict_label(self, texts, deterministic=True):
        r = self.predict_proba(texts, deterministic=deterministic).argmax(axis=1)
        return r

    @abstractmethod
    def get_pad_token(self):
        pass
    @abstractmethod
    def get_pad_token_id(self):
        pass