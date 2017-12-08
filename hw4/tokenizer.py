import sys
import csv
import pickle
from keras.preprocessing.text import Tokenizer

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

allSequences = getAllSequences('allSequences.txt')

#prepare tokenizer
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(allSequences)

#save tokenizer
with open('tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file)


