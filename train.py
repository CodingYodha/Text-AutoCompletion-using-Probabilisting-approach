from ngram_model import NGramModel

#initialize and train model
model = NGramModel(n=3)
model.train(corpus='textfile.txt')

#test predictions
print(model.predict("I enjoy"))

print(model.predict("science"))

print(model.predict("unknown text"))