from abc import ABC, abstractmethod
import numpy as np
class Detector(ABC):
    @abstractmethod
    def  __init__(self, metdata_only=False):
        pass
    @abstractmethod
    # out[0]: p(machine), out[1]: p(human)
    def predict_proba(self, texts, deterministic=True):
        """A sklearn style predict_proba function.

        Args:
            texts: A list of strings
            deterministic: If set to False, np and torch seeds are set for each document. Makes the output deterministic but removes the ability to process multiple documents at once. Defaults to True.

        Returns: 
            A numpy array of labels [(p_machine, p_human),...]
        """
        pass
    # returns 0 if machine is more likely, 1 if human more likely
    def predict_label(self, texts, deterministic=True):
        r = self.predict_proba(texts, deterministic=deterministic).argmax(axis=1)
        return r

    @abstractmethod
    def get_pad_token(self):
        """Returns the pad_token to be used by the explanation method to mask input as a string.
        """
        pass
    @abstractmethod
    def get_pad_token_id(self):
        """Returns the pad_token to be used by the explanation method to mask input as an id.
        """
        pass