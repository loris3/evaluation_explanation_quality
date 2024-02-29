from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from explainer_wrappers import Anchor_Explainer
from tqdm import tqdm
# create hybrid documents as proposed by Poerner et al.:
# sentence_tokenizer is common for all explanaiton methods so that the input to the detectors is the same. This is verified in an assert in the notebook
# word_tokenizer: use the same splitting/indexing method that the explanation method uses for easy calculation of the pointing game accuracy.
def hybrid(documents, labels, lenght = 6 , word_tokenizer=word_tokenize):
    hybrid_documents = []
    tokenized_hybrid_documents = []
    GT = []

    # poerner et al.: split into sentences
    tokenized_sentences =  [(x, label) for document, label in zip(documents, labels) for x in sent_tokenize(document)] # split then flatten into one global list of sentences
    # poerner et al.: shuffle
    np.random.seed(42)
    np.random.shuffle(tokenized_sentences)
    for i in range(0, len(tokenized_sentences), lenght): # TODO remove loop
        batch = tokenized_sentences[i:min(i+lenght, len(tokenized_sentences))]
        hybrid_tokenized_document = []
        hybrid_labels = []
        for sentence, label in batch:
            for word in word_tokenizer(sentence):
                hybrid_tokenized_document.append(word)
                hybrid_labels.append(label)
        hybrid_documents.append(" ".join([sentence for sentence,_ in batch]))
        tokenized_hybrid_documents.append(hybrid_tokenized_document)
        GT.append(np.array(hybrid_labels))
    return hybrid_documents, tokenized_hybrid_documents, GT


# as get_pointing_game_acc but returns the score for each document for a ttest
def get_pointing_game_scores(hybrid_documents, explainer, predictions_hybrid, GT):
    if(isinstance(explainer, Anchor_Explainer)):
        pointing_game_scores = []
        for document, gt in tqdm(list(zip(hybrid_documents, GT))):
            explanation = explainer.get_explanation_cached(document)
            positions = [x.idx for x in explainer.explainer.nlp(document)] # to resolve ids provided by explanation["positions"]
            assert len(positions) == len(gt)
            assert max(explanation["mean"]) == explanation["mean"][-1] # the next line assumes the longest anchor is the one with the highest precision
            pointing_game_scores.append(sum([int(gt[positions.index(p)] == explanation["prediction"]) for p in explanation["positions"]]) / len(explanation["positions"]))
    else:
        fi_scores = explainer.get_fi_scores_batch(hybrid_documents)
        
        # for each <explanation on a hybrid document> get the index of the top word by FI FOR THE PREDICTED CLASS.
        # in the notation of Poerner et al.:
        #                   ---------------------------
        #                   ---------rmax(X,phi)-------------------------------
        #                                   ---f(X)---           
        indices_top_word = [max(explanation[prediction], key=lambda x: x[1])[0] for explanation, prediction in zip(fi_scores, predictions_hybrid)] #  key=lambda x: x[1]: fi score for token; ...[0]: idx of token


        # now test for all rmax(X,phi) wether f(X,rmax(X,phi)) == f(X), i.e. the top word originates from "a document with the correct gold label"  
        # pointing_game_acc = ---------------------- number of hits ---------------------------------------------------------------------------------------------------   / possible hits
        #                         a list of booleans: 1 if the class of the top word (=prediction_hybrid by definition) matches the GT
        pointing_game_scores = [int(GT_doc[idx_top_word] == prediction_hybrid) for idx_top_word, GT_doc, prediction_hybrid in zip(indices_top_word,GT, predictions_hybrid)]
    return pointing_game_scores
