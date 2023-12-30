from fi_explainer import FI_Explainer
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import re

class LIME_Explainer(FI_Explainer):
    def __init__(self, detector, num_features=10, num_samples=1000):
        self.num_features = num_features
        self.num_samples = num_samples
        self.detector = detector #detector_class()
        self.explainer = LimeTextExplainer(class_names=["machine", "human"], bow=False, split_expression= r"\s",mask_string=self.detector.get_pad_token()) # TODO
   
        self.splitter = re.compile(r'(%s)|$' % self.explainer.split_expression) # for tokenize()
    def get_explanation(self, document):
        return self.explainer.explain_instance(document, self.detector.predict_proba, num_features=self.num_features, num_samples=self.num_samples, labels=[0,1]) # labels=0,1 as the detectors can technically be multi class
    def get_fi_scores(self, document):
        return self.get_explanation_cached(document).as_map()
    def get_fi_scores_batch(self, documents):
        return [self.get_fi_scores(document) for document in documents]
    
    def tokenize(self, document):
        return [s for s in self.splitter.split(document) if s and s!= " "] # as in LIME source

import shap
import transformers

class SHAP_Explainer(FI_Explainer):
    def __init__(self, detector):
        
        self.detector = detector #detector_class()

        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
                    "distilbert-base-uncased"
                ) # example from https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html
        
        # distilbert-base-uncased adds custom tokens to start and end, this makes indexing for the pointing game complex
        # --> don't add them
        self.custom_tokenizer = lambda s,return_offsets_mapping=True : self.tokenizer(s, return_offsets_mapping=return_offsets_mapping,add_special_tokens=False)


        self.masker = shap.maskers.Text(self.custom_tokenizer, mask_token=self.detector.get_pad_token())
        self.explainer = shap.Explainer(self.detector.predict_proba, masker=self.masker, output_names=["machine", "human"])


    def tokenize(self, document):
        return self.masker.data_transform(document)[0]


    def get_explanation(self, document):
        return self.explainer([document])
    def get_fi_scores(self, document):
        exp = self.get_explanation_cached(document)
        # reshape to match lime map
        values_machine = exp.values[:,:,0][0]
        values_human = exp.values[:,:,1][0]

        return {0 : list(enumerate(values_machine)), 1: list(enumerate(values_human))}
    def get_fi_scores_batch(self, documents):
        return [self.get_fi_scores(document) for document in documents]