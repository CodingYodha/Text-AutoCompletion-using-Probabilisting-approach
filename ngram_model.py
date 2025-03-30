class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(lamda: defaultdict(int))
        self.vocab = set()

    def train(self, corpus):
        #collect n-grams and vocabulary
        for sentence in corpus:
            tokens = preprocess(sentence)