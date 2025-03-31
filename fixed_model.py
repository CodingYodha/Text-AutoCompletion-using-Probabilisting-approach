# File: fixed_model.py
from collections import defaultdict
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        
    def train(self, corpus):
        for sentence in corpus:
            tokens = preprocess(sentence)
            if len(tokens) < self.n:
                continue  # Skip short sentences
                
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                self.ngrams[context][next_word] += 1
                
        # Apply smoothing
        self._apply_smoothing()
        
    def _apply_smoothing(self, alpha=0.1):
        vocab = list(self.vocab)
        for context in self.ngrams:
            total = sum(self.ngrams[context].values()) + alpha * len(vocab)
            for word in vocab:
                self.ngrams[context][word] = (self.ngrams[context].get(word, 0)) + alpha
                self.ngrams[context][word] /= total
                
    def predict(self, text, top_k=3):
        tokens = preprocess(text)
        context = tuple(tokens[-(self.n-1):]) if tokens else tuple()
        
        while len(context) >= 0:
            if context in self.ngrams:
                suggestions = sorted(self.ngrams[context].items(),
                                   key=lambda x: -x[1])[:top_k]
                return [word for word, _ in suggestions]
            if not context:
                break
            context = context[1:]  # Backoff
            
        # Fallback to unigrams
        unigrams = defaultdict(int)
        for ctx in self.ngrams.values():
            for word, count in ctx.items():
                unigrams[word] += count
        return sorted(unigrams.keys(), 
                     key=lambda x: -unigrams[x])[:top_k]

# Usage
with open('textfile.txt', 'r') as f:
    corpus = f.readlines()

model = NGramModel(n=3)
model.train(corpus)

print(model.predict("I enjoy"))     # ['reading']
print(model.predict("science"))     # ['fiction']
print(model.predict("unknown"))     # Top common words