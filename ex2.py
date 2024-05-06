"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""

# you can use these packages (uncomment as needed)
import pickle
import pandas as pd
import numpy as np
import os,time, re, sys, random, math, collections
import nltk

#static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Matan Leventer', 'id': '208447029', 'email': 'leventem@post.bgu.ac.il'}


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    with open(fn, 'r') as file:
        # Read the entire contents of the file
        corpus = file.read()
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    corpus = corpus.lower()
    corpus = re.sub(r'[^\w\s\n.]', '', corpus)
    corpus = re.sub(r'\s+', ' ', corpus)
    filtered_text = ' '.join([word for word in corpus.split() if word  not in stop_words])
    sentences = [i.strip() for i in filtered_text.split('.')]
    #TODO
    return sentences

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    #
    file = open(fn, "rb")
    sg_model = pickle.load(file)
    file.close()

    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):

        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context #the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold #ignore low frequency words (appearing under the threshold)
        self.model_dict = collections.Counter(' '.join(sentences).split())
        self.T = None
        self.C = None
        self.V = None

        for word, count in list(self.model_dict.items()):
            if count < word_count_threshold:
                del self.model_dict[word]

        self.len_model_dict = len(self.model_dict)
        self.index_word={}
        for idx,w in enumerate(self.model_dict):
            self.index_word[w]=idx

        # Tips:
        # 1. It is recommended to create a word:count dictionary
        # 2. It is recommended to create a word-index map

    # TODO
    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        try:
            w1 = w1.lower()
            w2 = w2.lower()


            A=self.V[:,self.index_word[w1]]
            B=self.V[:,self.index_word[w2]]

            sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        except:
            return 0.0
        # sim  = 0.0 # default
        #TODO

        return sim # default

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        w = w.lower()
        if w not in self.model_dict:
            return []
        list_result=[]
        list_word=[]
        for word in self.index_word:
            if word == w:
                continue
            list_result.append(self.compute_similarity(w,word))
            list_word.append(word)
        list_result_sorted = sorted(list_result,reverse=True)[:n]
        list_result_word= [list_word[list_result.index(i)] for i in list_result_sorted]
        return list_result_word


    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """


        vocab_size = self.len_model_dict
        ... #todo: set to be the number of words in the model (how? how many, indeed?)
        T = np.random.rand(self.d, vocab_size) # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words
        pos_list=[]
        for sentance in self.sentences:
            pos_lst = list(nltk.skipgrams((sentance.split()), int((self.context)/ 2), 1))

            pos_lst +=[(tup[1],tup[0]) for tup in pos_lst]
            pos_list += pos_lst

        dict_pos = self.create_dict_neg_pos(pos_list)

        neg_lst = []
        for word in self.model_dict:
            try:
                for _ in range(self.neg_samples * len(dict_pos[word])):
                    neg_lst += [(word, random.choices(list(self.model_dict.keys()), weights=list(self.model_dict.values()))[0])]
            except:
                continue

        dict_neg = self.create_dict_neg_pos(neg_lst)
        prev_loss=0
        c=0
        for epoch in range(epochs):
            loss = 0
            for word,index in self.index_word.items():
                if word not in dict_pos:
                    continue
                learning_word = np.zeros(vocab_size)
                learning_word_t = np.zeros(vocab_size)
                learning_word_t[index] =1
                y=[1]*len((dict_pos[word])) + [0]*len((dict_neg[word]))
                loss_p = []
                index_p = []
                t = T[:,index]

                for pos in (dict_pos[word]):
                    c_pos= C[self.index_word[pos],:]
                    x_pos = np.dot(c_pos,t)
                    loss_p.append(sigmoid(x_pos))
                    index_p.append(self.index_word[pos])
                    learning_word[self.index_word[pos]]+=1


                for neg in (dict_neg[word]):
                    c_neg = C[self.index_word[neg], :]
                    x_neg = np.dot(c_neg, t)
                    loss_p.append(sigmoid(x_neg))
                    index_p.append(self.index_word[neg])
                    learning_word[self.index_word[neg]]+=1


                res= np.zeros(vocab_size)
                mask = np.zeros(vocab_size)

                # Compute the loss and the gradient of the loss
                loss += np.mean(self.log_loss(y, loss_p))
                mone = (np.subtract(loss_p,y))

                for loc,idx in enumerate(index_p):
                    res[idx]+= mone[loc]

                for loc,v in enumerate(res):
                    if (learning_word[loc]) == 0:
                        mask[loc] = 0.0
                    else:
                        mask[loc] = v/(learning_word[loc])

                # Compute the gradient of the loss with respect to the weights

                dW_C = np.dot(t[:, None],mask[:, None].transpose()).transpose()

                dW_T = np.dot(np.vstack(learning_word_t),np.dot(C.transpose(),mask)[:, None].transpose()).transpose()

                C -= step_size * dW_C
                T -= step_size * dW_T

            if loss/self.len_model_dict>prev_loss:
                c+=1
                prev_loss=loss/self.len_model_dict
            else:
                c=0
                prev_loss = loss / self.len_model_dict
            if c==early_stopping:
                break

        #tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.

        # TODO
        self.T = T
        self.C = C
        if model_path is not None:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
        return T, C

    def log_loss(self, y, p):
        ones = np.ones(len(y))
        loss = np.mean(-((y * np.log(p)) + np.subtract(ones, y) * np.log(np.subtract(ones, p))))
        return loss

    def create_dict_neg_pos(self,list_p_n):
        dict_pos_neg= {}
        for word,word_pos in list_p_n:
            if word in self.model_dict and word_pos in self.model_dict:
                if word in dict_pos_neg :
                    dict_pos_neg[word].append(word_pos)
                else:
                    dict_pos_neg[word] = [word_pos]
        return dict_pos_neg


    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        # TODO
        if combo==0:
            embedding_matrix = T
        elif combo == 1:
            embedding_matrix = C
        elif combo == 2:
            embedding_matrix = np.mean((T,C.T), axis=0)
        elif combo == 3:
            embedding_matrix = np.sum((T,C.T), axis=0)
        elif combo == 4:
            embedding_matrix = np.concatenate((T, C.T), axis=0)
        else:
            raise ValueError("Invalid combo value")

        self.V = embedding_matrix
        if model_path is not None:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
        return embedding_matrix


    def find_analogy(self, w1,w2,w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        #TODO
        try:
            w1 = w1.lower()
            w2 = w2.lower()
            w3 = w3.lower()

            _ = self.V[:,self.index_word[w1]]
            _ = self.V[:,self.index_word[w2]]
            _ = self.V[:,self.index_word[w3]]
        except KeyError as e:
            return ''



        for word in self.index_word:
            if word == w1 or word == w2 or word == w3:
                continue
            if self.test_analogy(w1,w2,w3,word,3):
                return word

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        # TODO
        # Retrieve the vector representations of the four words
        try:

            w1 = w1.lower()
            w2 = w2.lower()
            w3 = w3.lower()
            w4 = w4.lower()

            v1 = self.V[:,self.index_word[w1]]
            v2 = self.V[:,self.index_word[w2]]
            v3 = self.V[:,self.index_word[w3]]
            v4 = self.V[:,self.index_word[w4]]
        except KeyError as e:
            return False

        target = v1 - v2 + v3

        similarity_scores = np.dot(self.V.transpose(), target) / (np.linalg.norm(self.V.transpose(), axis=1) * np.linalg.norm(target))

        top_n_indices = np.argsort(similarity_scores)[-n:]

        if v4 in self.V.transpose()[top_n_indices]:
            return True
        else:
            return False

