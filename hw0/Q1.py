import sys

inputFile = open(sys.argv[1], 'r')
outputFile = open('Q1.txt', 'w')
s = inputFile.read()
s = s.split()
index = 0
result = []
for i in s:
	find = 0
	for j in result:
		if j[0] == i:
			j[1] = j[1] + 1
			find = 1
	if find != 1:
		result.append([i,1])

for i in result:
	outputFile.write(i[0])
	outputFile.write(' ')
	outputFile.write(str(index))
	outputFile.write(' ')
	outputFile.write(str(i[1]))
	outputFile.write('\n')
	index = index + 1

inputFile.close()
outputFile.close()
