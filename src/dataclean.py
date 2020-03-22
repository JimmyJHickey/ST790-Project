import pandas as pd
train = pd.read_csv('../Kannada-MNIST/train.csv')
test = pd.read_csv('../Kannada-MNIST/test.csv')
print(train)
trainY = train[train.columns[0]]
print(trainY)
trainX = train[train.columns[1:]]
print(trainX)
testY = test[test.columns[0]]
print(testY)
testX = test[test.columns[1:]]
print(testX)