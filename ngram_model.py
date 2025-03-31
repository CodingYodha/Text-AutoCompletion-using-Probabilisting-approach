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

        #add laplace smoothing
        self._apply_smoothing()

    def _apply_smoothing(self, alpha=0.1):
        vocab_size = len(self.vocab)
        for context in self.ngrams:
            total = sum(self.ngrams[context].values())+alpha*vocab_size
            for word in self.vocab:
                self.ngrams[context][word] = (self.ngrams[context].get(word, 0) + alpha)/total
            
        