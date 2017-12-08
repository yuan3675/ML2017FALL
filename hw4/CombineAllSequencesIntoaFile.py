import sys
import csv

temp = []
decrement = True
maxInt = sys.maxsize
testName = sys.argv[3]
trainLabelName = sys.argv[1]
trainUnlabelName = sys.argv[2]

#unlabel data size too large, do some processes        
while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

#read sequences from testing data                
csv.field_size_limit(maxInt)
with open(testName, encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        row.pop(0)
        sentence = " ".join(row)
        temp.append(sentence)
csvfile.close()
temp.pop(0)

#read sequences from labeled training data        
with open(trainLabelName, encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        label = row.pop(0)
        row.pop(0)
        sentence = " ".join(row)
        temp.append(sentence)
csvfile.close()

#read sequences from unlabeled training data
with open(trainUnlabelName, encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        sentence = " ".join(row)
        temp.append(sentence)
csvfile.close()

#write all sequences into a file
with open(sys.argv[4], 'w', encoding='utf-8', newline='') as file:
    for i in temp:
        file.write(i + '\n')
    csvfile.close()
