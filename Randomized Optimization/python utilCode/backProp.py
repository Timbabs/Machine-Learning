
import numpy as np
import datasetSpliter as splitter
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time

testAccuracy = []
trainAccuracy = []
iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

data = splitter.trainData
trainData = np.ndarray(shape=(398, 30), dtype=float, order='F')
for i in range(0, 30):
    trainData[:, i] = data[:, i]
trainTarget = data[:, 30]

data = splitter.testData
testData = np.ndarray(shape=(171, 30), dtype=float, order='F')
for i in range(0, 30):
    testData[:, i] = data[:, i]
testTarget = data[:, 30]


scaler = StandardScaler()

# Fit only to the training data
scaler.fit(trainData)

# Now apply the transformations to the data:
trainData = scaler.transform(trainData)
testData = scaler.transform(testData)

trainTime = []
for i in iterations:
    print('processing iteration: ' + str(i))
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=2000)

    start_time = time.clock()
    for j in range(0, i, 1):
        mlp.fit(trainData, trainTarget)
    trainTime.append((time.clock() - start_time)/i)

    y_pred = mlp.predict(trainData)
    trainAccuracy.append(metrics.accuracy_score(trainTarget, y_pred))

    y_pred = mlp.predict(testData)
    testAccuracy.append(metrics.accuracy_score(testTarget, y_pred))
sourceFileTrain = '../Assignment2/ABAGAIL/src/opt/test/myTests/outputData/'+ 'backpropTrain.txt'
sourceFileTest = '../Assignment2/ABAGAIL/src/opt/test/myTests/outputData/'+ 'backpropTest.txt'
with open(sourceFileTrain, "w+") as f:
    [f.write(str(value)+'\n') for value in trainAccuracy]
f.close()
with open(sourceFileTest, "w+") as f:
    [f.write(str(value)+'\n') for value in testAccuracy]
f.close()
print(np.mean(trainTime))
