from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


# create hybrid documents as proposed by Poerner et al.:
# sentence_tokenizer is common for all explanaiton methods so that the input to the detectors is the same
# word_tokenizer: use the same splitting/indexing method that the explanation method uses for easy calculation of the pointing game accuracy.
def hybrid(documents, labels, lenght = 10 , word_tokenizer=word_tokenize):
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
     #   h = []
   #     hybrid_X = []
        hybrid_labels = []
    #    print(batch)
        for sentence, label in batch:
           # print("sent",sentence)
            for word in word_tokenizer(sentence):
              #  h.append(sentence)
                hybrid_tokenized_document.append(word)
              #  hybrid_X.append(word)
                hybrid_labels.append(label)
       # print("hybrid_tokenized_document",hybrid_tokenized_document)
        hybrid_documents.append(" ".join([sentence for sentence,_ in batch]))
        tokenized_hybrid_documents.append(hybrid_tokenized_document)
        GT.append(np.array(hybrid_labels))

    # print("hybrid_documents",hybrid_documents)
    # print("tokenized_hybrid_documents",tokenized_hybrid_documents)
    # print("GT", GT)
    return hybrid_documents, tokenized_hybrid_documents, GT

def get_pointing_game_acc(hybrid_documents, explainer, predictions_hybrid, GT):
    
    fi_scores = explainer.get_fi_scores_batch(hybrid_documents)
    
    # for each <explanation on a hybrid document> get the index of the top word by FI FOR THE PREDICTED CLASS.
    # in the notation of Poerner et al.:
    #                   ---------------------------
    #                   ---------rmax(X,phi)-------------------------------
    #                                   ---f(X)---           
    indices_top_word = [max(explanation[prediction], key=lambda x: x[1])[0] for explanation, prediction in zip(fi_scores, predictions_hybrid)] 


    # now test for all rmax(X,phi) wether f(X,rmax(X,phi)) == f(X), i.e. the top word originates from "a document with the correct gold label"  
    # pointing_game_acc = ---------------------- number of hits ---------------------------------------------------------------------------------------------------   / possible hits
    #                         a list of booleans: 1 if the class of the top word (=prediction_hybrid by definition) matches the GT
    pointing_game_acc = sum([GT_doc[idx_top_word] == prediction_hybrid for idx_top_word, GT_doc, prediction_hybrid in zip(indices_top_word,GT, predictions_hybrid)]) / len(hybrid_documents)
    return pointing_game_acc



