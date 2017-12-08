import sys
import csv
import logging
from gensim import corpora
from collections import defaultdict
from gensim.models import word2vec

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def getAllSequences(filename):
    temp = []
    decrement = True
    maxInt = sys.maxsize
        
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True
                
    csv.field_size_limit(maxInt)

    with open(filename, encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            sentence = " ".join(row)
            temp.append(sentence)
    csvfile.close()

    return(temp)

#get all sequences
allSequences = getAllSequences('allSequences.txt')

#remove common words and tokenize
output = open('newAllSequences.txt', 'w', encoding='utf-8')
stopList = set('for a of the and to in .'.split())
for sequence in allSequences:
    words = sequence.split()
    for word in words:
        if word not in stopList:
            output.write(word + ' ')

sentences = word2vec.Text8Corpus('newAllSequences.txt')
model = word2vec.Word2Vec(sentences, size = 128)

# Save model
model.save('allSequences.bin')


# To load a model.
# model = word2vec.Word2Vec.load("your_model.bin")




