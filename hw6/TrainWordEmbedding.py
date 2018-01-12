#encoding=utf-8
from gensim import corpora
from gensim.models import word2vec
from collections import defaultdict
import jieba, re

jieba.set_dictionary("C:\\Users\\Yuan\\Desktop\\dict.txt.big")

file = open('C:\\Users\\Yuan\\Desktop\\all_sents.txt', 'r', encoding = 'utf-8')

output = open('C:\\Users\\Yuan\\Desktop\\allWords.txt', 'w', encoding = 'utf-8')

for line in file.readlines():
    words = jieba.cut(line)    
    for word in words:
        output.write(word + ' ')
    output.write('\n')
output.close()
file.close()

trainingData = word2vec.Text8Corpus('C:\\Users\\Yuan\\Desktop\\allWords.txt')
model1 = word2vec.Word2Vec(trainingData, size = 128, min_count = 3000)
model1.save('over6000Words.bin')
model2 = word2vec.Word2Vec(trainingData, size = 128, window = 10, min_count = 3)
model2.save('allWords.bin')

