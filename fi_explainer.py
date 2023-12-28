from abc import ABC, abstractmethod
import pickle
import os
import hashlib
sha256 = hashlib.sha256()



class FI_Explainer(ABC):
    @abstractmethod
    def  __init__(self):
        pass

    # return a list of (word,feature-importance) tuples
    @abstractmethod
    def get_fi_scores(self, document):
        pass
    
    def get_explanations_cached(self, documents, cache_dir="./explanation_cache"):
        return [self.get_explanation_cached(document, cache_dir) for document in documents]


    def get_explanation_cached(self, document, cache_dir="./explanation_cache"):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
      #  print(type(document), document)
        sha256.update(document.encode('utf-8'))
        exp_hash = sha256.hexdigest()
        path = os.path.join(cache_dir,str(exp_hash)+".pkl")
        if os.path.isfile(path):
            return pickle.load(open(path,'rb'))
        else:
            print("regen", document)
            e = self.get_explanation(document)

            pickle.dump(e, open(path,'wb'))
            return e
            