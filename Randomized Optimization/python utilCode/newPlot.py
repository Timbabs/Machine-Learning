import matplotlib.pyplot as plt
import numpy as np


iterations = np.arange(0, 1001, 20)
algorithms = ['RHC', 'SA', 'GA']
N = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
figure1 = 1
figure2 = 2
fig1 = plt.figure(figure1)
fig2 = plt.figure(figure2)
for algo in algorithms:
    sourceFile = '../Assignment2/ABAGAIL//src/opt/test/optData/TST_variedIteration/' +algo + '.txt'
    with open(sourceFile) as f:
        data = f.read()
    fitness = [float(value) for value in data.split("\n") if value]
    plt.figure(figure1)
    plt.plot(iterations, fitness, label=algo, lw=2.0)
    plt.axis([np.min(iterations), np.max(iterations), 0, 0.2])
    plt.xlabel("iterations")
    plt.ylabel("Fitness")
    plt.title("Travelling Salesman Test")
    plt.legend(loc='lower right')

    sourceFile = '../Assignment2/ABAGAIL//src/opt/test/optData/TST_varied_N/' + algo + '.txt'
    with open(sourceFile) as f:
        data = f.read()
    fitness = [float(value) for value in data.split("\n") if value]
    plt.figure(figure2)
    plt.plot(N, fitness, label=algo, lw=2.0)
    #plt.axis([np.min(iterations), np.max(iterations), 0, 0.4])
    plt.xlabel("N")
    plt.ylabel("Fitness")
    plt.title("Travelling Salesman Test")
    plt.legend(loc='upper right')

fig1.savefig('../Assignment2/ABAGAIL/src/opt/test/optGraph/TST_variedIteration.png', dpi=300)
fig2.savefig('../Assignment2/ABAGAIL/src/opt/test/optGraph/TST_varied_N.png', dpi=300)


figure1 = figure1 + 2
figure2 = figure2 + 2
fig1 = plt.figure(figure1)
fig2 = plt.figure(figure2)
newIterations = np.arange(0, 4501, 125)
algorithms = ['RHC', 'SA', 'GA']
for algo in algorithms:
    sourceFile = '../Assignment2/ABAGAIL//src/opt/test/optData/FF_variedIteration/' +algo + '.txt'
    with open(sourceFile) as f:
        data = f.read()
    fitness = [float(value) for value in data.split("\n") if value]
    plt.figure(figure1)
    plt.plot(newIterations, fitness, label=algo, lw=2.0)
    plt.axis([np.min(iterations), np.max(iterations), 40, 100])
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.title("Flip Flop Test")
    plt.legend(loc='lower right')

    sourceFile = '../Assignment2/ABAGAIL//src/opt/test/optData/FF_varied_N/' + algo + '.txt'
    with open(sourceFile) as f:
        data = f.read()
    fitness = [float(value) for value in data.split("\n") if value]
    plt.figure(figure2)
    plt.plot(N, fitness, label=algo, lw=2.0)
    # plt.axis([np.min(iterations), np.max(iterations), 0.01, 0.4])
    plt.xlabel("N")
    plt.ylabel("fitness")
    plt.title("Flip Flop Test")
    plt.legend(loc='lower right')

fig1.savefig('../Assignment2/ABAGAIL/src/opt/test/optGraph/FF_variedIteration.png', dpi=300)
fig2.savefig('../Assignment2/ABAGAIL/src/opt/test/optGraph/FF_varied_N.png', dpi=300)


figure1 = figure1 + 2
figure2 = figure2 + 2
fig1 = plt.figure(figure1)
fig2 = plt.figure(figure2)
algorithms = ['SA', 'RHC', 'GA']
for algo in algorithms:
    sourceFile = '../Assignment2/ABAGAIL//src/opt/test/optData/CO_variedIteration/' +algo + '.txt'
    with open(sourceFile) as f:
        data = f.read()
    fitness = [float(value) for value in data.split("\n") if value]
    plt.figure(figure1)
    plt.plot(iterations, fitness, label=algo, lw=2.0)
    #plt.axis([np.min(iterations), np.max(iterations), 0, 100])
    plt.xlabel("iterations")
    plt.ylabel("fitness")
    plt.title("Counting Ones Test")
    plt.legend(loc='lower right')

    sourceFile = '../Assignment2/ABAGAIL//src/opt/test/optData/CO_varied_N/' + algo + '.txt'
    with open(sourceFile) as f:
        data = f.read()
    fitness = [float(value) for value in data.split("\n") if value]
    plt.figure(figure2)
    plt.plot(N, fitness, label=algo, lw=2.0)
    # plt.axis([np.min(iterations), np.max(iterations), 0.01, 0.4])
    plt.xlabel("N (Number of vertices)")
    plt.ylabel("fitness")
    plt.title("Counting Ones Test")
    plt.legend(loc='lower right')
fig1.savefig('../Assignment2/ABAGAIL/src/opt/test/optGraph/CO_variedIteration.png', dpi=300)
fig2.savefig('../Assignment2/ABAGAIL/src/opt/test/optGraph/CO_varied_N.png', dpi=300)

plt.show()
