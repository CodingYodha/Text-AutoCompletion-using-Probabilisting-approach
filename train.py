from ngram_model import NGramModel

with open('textfile.txt', 'r', encoding='utf-8') as file:
    corpus_text = file.read()

#initialize and train model
model = NGramModel(n=3)
model.train(corpus=corpus_text)

#test predictions
print(model.predict("I enjoy"))

print(model.predict("science"))

print(model.predict("unknown text"))