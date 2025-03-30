import re
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) #keep words/numbers/whitespace
    return text.split()

#test preprocessing
print(preprocess("Hello! How's the weather?"))