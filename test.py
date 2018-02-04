import pandas as pd
from six.moves import cPickle
from scipy.sparse import hstack

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

models = []
for i in classes:
    with open(i + "_model.pkl", 'rb') as f:
        models.append(cPickle.load(f))


with open("word_vectorizer.pkl", 'rb') as f:
    word = cPickle.load(f)

with open("char_vectorizer.pkl", 'rb') as f:
    char = cPickle.load(f)

s = ["you fool dumb basterds"]
wf = word.transform(s)
cg = char.transform(s)

traife = hstack([cg, wf])

pred = []
for i, c in enumerate(classes):
    pred.append(models[i].predict_proba(traife)[:, 1])

for i, c in enumerate(classes):
    print("Probability of ", c, " is : ", pred[i][0])
