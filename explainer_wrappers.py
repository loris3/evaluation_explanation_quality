from fi_explainer import FI_Explainer
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer, TextDomainMapper
import re
import os
import numpy as np
from lime.explanation import id_generator
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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
    
    # shortened version of as_html() without the barplots for TextExplainer
    # always explains with machine as reference (blue - orange + FI for machine)
    def get_highlighted_text_HTML(self, document):
            explanation = self.get_explanation_cached(document)
            bundle = open(os.path.join("lime/lime/bundle.js"),
                            encoding="utf8").read()
            out = u'''<html>
                    <meta http-equiv="content-type" content="text/html; charset=UTF8">
                    <head><script>%s </script></head><body>''' % bundle
            random_id = id_generator(size=15, random_state=check_random_state(self.explainer.random_state)) # done in LIME
            out += u'''
            <div class="lime top_div" id="top_div%s"></div>
            ''' % random_id

            exp_js = 'var exp_div;\n        var exp = new lime.Explanation(["machine", "human"]);\n' # only effects number of labels so that color is static
            raw_js = '''var raw_div = top_div.append('div');'''
            raw_js += explanation.domain_mapper.visualize_instance_html(
                            explanation.local_exp[0],
                            0,
                            'raw_div',
                            'exp')
            out += u'''
            <script>
            var top_div = d3.select('#top_div%s').classed('lime top_div', true);
            %s
            %s
            %s
            %s
            </script>
            ''' % (random_id, "", "", exp_js, raw_js)
            out += u'</body></html>'
            return out
     # adapts LIME internal .as_pyplot() and returns two HTML strings
    def get_barplots_HTML(self, document):
        plt.ioff()

        explanation = self.get_explanation_cached(document)
        fig = plt.figure()
        exp = explanation.as_list(label=0)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['orange' if x > 0 else 'blue' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)

        all_fi_scores = [x[1] for xs in explanation.as_map().values() for x in xs]

        plt.xlim(min(all_fi_scores), max(all_fi_scores))
        plt.yticks(pos, names)

        plt.suptitle("let f(x) = machine",fontsize=16, y=1)
        plt.title("decreases confidence | increases confidence")
        # https://stackoverflow.com/questions/48717794/matplotlib-embed-figures-in-auto-generated-html
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        barplot_machine = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)



        exp = explanation.as_list(label=1)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)


        plt.xlim(min(all_fi_scores), max(all_fi_scores))
        plt.yticks(pos, names)

        title = 'Local explanation for class %s' % explanation.class_names[1]

        plt.suptitle("let f(x) = human",fontsize=16, y=1)
        plt.title("decreases confidence | increases confidence")

        # https://stackoverflow.com/questions/48717794/matplotlib-embed-figures-in-auto-generated-html
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png');
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        barplot_human = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        
        return barplot_machine, barplot_human


    def get_vanilla_visualization_HTML(self, document):
        explanation = self.get_explanation_cached(document )

        # LIME shows the barplots in order of "top_labels", i.e. machine first if prediction is machine, human first if prediction is human
        # orange is positive FI scores for top_lablel, blue negative FI scores
        # This is desirable for multiclass problems (when using top_labels=n)
        # But really confusing here. 
        # labels=[0,1]  --> Fix order: always show machine first, always use machine as reference for text plot
        return explanation.as_html(labels=[0,1], predict_proba=False) 
    def as_list(self, exp, label=0):
        return exp.as_list(label=label)

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
    
    def get_vanilla_visualization_HTML(self, document):
        explanation = self.get_explanation_cached(document)
        return shap.plots.text(explanation, display=False)
    def as_list(self, exp, label=0):
        return list(zip(exp.data[0], exp.values[0,:,label]))
    # TODO duplicate code
    def get_barplots_HTML(self, document):
        plt.ioff()
        fig = plt.figure()
        _ = shap.plots.bar(self.get_explanation_cached(document)[0,:,0], show=False)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png');
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        barplot_machine = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

        fig = plt.figure()
        _ = shap.plots.bar(self.get_explanation_cached(document)[0,:,1], show=False)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png');
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        barplot_human = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

        return barplot_machine, barplot_human
    def get_highlighted_text_HTML(self, document):

        return shap.plots.text(self.get_explanation_cached(document),display=False)
