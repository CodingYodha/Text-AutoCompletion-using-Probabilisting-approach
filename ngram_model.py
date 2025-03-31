import re
from collections import defaultdict
from preprocess import preprocess

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, corpus):
        #collect n-grams and vocabulary
        for sentence in corpus:
            tokens = preprocess(sentence)
            self.vocab.update(tokens)
            for i in range(len(tokens)-self.n+1):
                context = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                self.ngrams[context][next_word]+=1
            
        