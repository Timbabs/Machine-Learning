import matplotlib.pyplot as plt

testAccuracy = []
trainAccuracy = []
iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

trainSize = [39, 79, 119, 159, 199, 239, 279, 319, 359]

varied = 'variedIteration'
algorithms = ['RHC', 'SA', 'GA']
dataset = ['train', 'test']
figure = 0

fig1 = plt.figure(1)
fig1.suptitle("Train Accuracy vs. Iterations (Overall)")
fig2 = plt.figure(2)
fig2.suptitle("Test Accuracy vs. Iterations (Overall)")

for algo in algorithms:
    sourceFileTrain = '../Assignment2/ABAGAIL/src/opt/test/myTests/outputData/' + str(varied) + '/' + str(
        algo) + '/' + str(algo) + '_' + '' + 'trainAccuracy_70.txt'
    sourceFileTest = '../Assignment2/ABAGAIL/src/opt/test/myTests/outputData/' + str(varied) + '/' + str(
        algo) + '/' + str(algo) + '_' + '' + 'testAccuracy_70.txt'
    with open(sourceFileTrain) as f:
        data = f.read()
    accuracy = [float(value) for value in data.split("\n") if value]

    plt.figure(1)
    plt.plot(iterations, accuracy, label=algo, lw=2.0)
   # plt.axis([np.min(iterations), np.max(iterations), 0, 100])



    with open(sourceFileTest) as f:
        data = f.read()
    accuracy = [float(value) for value in data.split("\n") if value]

    plt.figure(2)
    plt.plot(iterations, accuracy, label=algo, lw=2.0)
    #plt.axis([np.min(iterations), np.max(iterations), 0, 100])


sourceFileTrain = '../Assignment2/ABAGAIL/src/opt/test/myTests/outputData/' + 'backpropTrain.txt'
sourceFileTest = '../Assignment2/ABAGAIL/src/opt/test/myTests/outputData/' + 'backpropTest.txt'
with open(sourceFileTrain) as f:
    data = f.read()
trainAccuracy = [float(value)*100 for value in data.split("\n") if value]
plt.figure(1)
plt.plot(iterations, trainAccuracy, label="backpropagation", lw=2.0)

plt.xlabel("iteration")
plt.ylabel("Classification Accuracy")
plt.legend(loc='lower right')

with open(sourceFileTest) as f:
    data = f.read()
testAccuracy = [float(value)*100 for value in data.split("\n") if value]

plt.figure(2)
plt.plot(iterations, testAccuracy, label="backpropagation", lw=2.0)

plt.xlabel("iteration")
plt.ylabel("Classification Accuracy")
plt.legend(loc='lower right')

fig1.savefig(
    '../Assignment2/ABAGAIL/src/opt/test/myTests/outputGraphs/' + "trainAccuracy" + '_' + 'overall' + '_' + "_plot" + '.png',
    dpi=300)
fig2.savefig(
    '../Assignment2/ABAGAIL/src/opt/test/myTests/outputGraphs/' + 'testAccuracy' + '_' + 'overall' + '_' + "_plot" + '.png',
    dpi=300)
# plt.show()






