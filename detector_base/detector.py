from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def  __init__(self):
        pass
    @abstractmethod
    def detect(self, text):
        pass