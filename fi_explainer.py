from abc import ABC, abstractmethod
import pickle
import os
import hashlib




class FI_Explainer(ABC):
    @abstractmethod
    def  __init__(self, detector):
        pass

    # return a list of (word,feature-importance) tuples
    @abstractmethod
    # fill: for e.g. lime, only the top_10 features by fi are returned by default. Setting fill to True returns all features (0=word_0, .. n=word_n) and fills with 0. Useful for slicing explanations 
    def get_fi_scores(self, document, fill=False):
        pass

    @abstractmethod
    def tokenize(self, document, collapse_whitespace=True):
        pass
    @abstractmethod
    def get_vanilla_visualization_HTML(self, document):
        pass
    
    def get_explanations_cached(self, documents, cache_dir="./explanation_cache"):
        return [self.get_explanation_cached(document, cache_dir) for document in documents]

    def delete_cached_explanation(self, document, cache_dir="./explanation_cache"):
        sha256 = hashlib.sha256()
        sha256.update(document.encode('utf-8'))
        exp_hash = sha256.hexdigest()
        path = os.path.join(cache_dir,str(exp_hash)+"_"+self.__class__.__name__+"_"+self.detector.__class__.__name__+".pkl")
       # print(path)
        if os.path.isfile(path):
            os.remove(path)
    def does_explanation_exist(self, document, cache_dir="./explanation_cache"):
        sha256 = hashlib.sha256()
        sha256.update(document.encode('utf-8'))
        exp_hash = sha256.hexdigest()
        path = os.path.join(cache_dir,str(exp_hash)+"_"+self.__class__.__name__+"_"+self.detector.__class__.__name__+".pkl")
        return os.path.isfile(path)
    def get_explanation_cached(self, document, cache_dir="./explanation_cache"):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
      #  print(type(document), document)
        sha256 = hashlib.sha256()
        sha256.update(document.encode('utf-8'))
        exp_hash = sha256.hexdigest()
        path = os.path.join(cache_dir,str(exp_hash)+"_"+self.__class__.__name__+"_"+self.detector.__class__.__name__+".pkl")
       # print(path)
        if os.path.isfile(path):
            return pickle.load(open(path,'rb'))
        else:
            print("regen", document)
            e = self.get_explanation(document)

            pickle.dump(e, open(path,'wb'))
            return e
    @abstractmethod     
    def as_list(self, exp, label=0):
        pass
    @abstractmethod
    def get_barplots_HTML(self, document):
        pass
    @abstractmethod
    def get_vanilla_visualization_HTML(self, document):
        pass
    @abstractmethod
    def get_highlighted_text_HTML(self, document):
        pass
    @abstractmethod
    def untokenize(self, tokens):
        pass