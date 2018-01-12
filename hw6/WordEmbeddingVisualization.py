from gensim import corpora
from gensim.models import word2vec
from sklearn.manifold import TSNE
import re, adjustText, csv
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import findfont, FontProperties

#print(mp.matplotlib_fname())


def plot(Xs, Ys, Texts):
    plt.plot(Xs, Ys, 'o')
    texts = [plt.text(X, Y, Text) for X, Y, Text in zip(Xs, Ys, Texts)]
    plt.title(str(adjustText.adjust_text(texts, Xs, Ys, arrowprops = dict(arrowstyle = '->', color = 'red'))))
    plt.show()
  
model = word2vec.Word2Vec.load('allWords.bin')
drawModel = word2vec.Word2Vec.load('over5000Words.bin')

#recording the word index which appears over 3000~6000 times
index = []
for drawWord in drawModel.wv.vocab:
    counter = 0
    for word in model.wv.vocab:
        if drawWord == word:
            index.append(counter)
            break
        counter = counter + 1
"""
#reading the TSNE vector
newWordVec = []
with open('TSNEVector2.csv', 'r', encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        newWordVec.append([float(row[0]), float(row[1])])
"""
#take out all word vectors
wordvec = []
for word in model.wv.vocab:
    wordvec.append(model.wv[word])

#training TSNE
print('Doing TSNE......')
newWordVec = TSNE(n_components = 2).fit_transform(wordvec)
newWordVec = np.array(newWordVec)

#saving TSNE training result
print('Saving result......')
with open('TSNEVector2.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in newWordVec:
        writer.writerow([i[0], i[1]])
csvfile.close()

#take out the word vectors which appears over 3000~6000 times
vec = []
for i in index:
    a = []
    a.append(newWordVec[i][0])
    a.append(newWordVec[i][1])
    vec.append(a)
vec = np.array(vec)

#Visualize texts
plot(vec[:, 0], vec[:, 1], drawModel.wv.vocab)

