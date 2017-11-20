import sys
import FileReader

reader = FileReader.FileReader()
X_train = reader.getX_Train(sys.argv[1])
Y_train = reader.getY_Train()
X_test = reader.getX_Test(sys.argv[2])

