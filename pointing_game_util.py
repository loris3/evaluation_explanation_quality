from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


# create hybrid documents as proposed by Poerner et al.:
# word_tokenizer: use the same that the explanation method uses for easy evaluation. The detectors are fed the orgiginal documents.
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
        h = []
        hybrid_X = []
        hybrid_labels = []
    #    print(batch)
        for sentence, label in batch:
            for word in word_tokenizer(sentence):
                h.append(sentence)
                hybrid_tokenized_document.append(word)
                hybrid_X.append(word)
                hybrid_labels.append(label)

        hybrid_documents.append(" ".join([sentence for sentence,_ in batch]))
        tokenized_hybrid_documents.append(hybrid_tokenized_document)
        GT.append(np.array(hybrid_labels))

    # print("hybrid_documents",hybrid_documents)
    # print("tokenized_hybrid_documents",tokenized_hybrid_documents)
    # print("GT", GT)
    return hybrid_documents, tokenized_hybrid_documents, GT

def get_pointing_game_acc(hybrid_documents, explainer, predictions_hybrid, GT):
    
    fi_scores = explainer.get_fi_scores_batch(hybrid_documents)
    
    indices_top_word = [max(explanation[prediction], key=lambda x: x[1])[0] for explanation, prediction in zip(fi_scores, predictions_hybrid)]
    pointing_game_acc = sum([GT_doc[idx_top_word] == prediction_hybrid for idx_top_word, GT_doc, prediction_hybrid in zip(indices_top_word,GT, predictions_hybrid)]) / len(hybrid_documents)
    return pointing_game_acc



