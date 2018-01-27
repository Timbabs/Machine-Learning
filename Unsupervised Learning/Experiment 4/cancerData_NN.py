from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

cancer = load_breast_cancer()
y = cancer.target

# Experiment1.
for randState in [2]:
    feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
    wisconsin = pd.read_csv('../utils/data.csv', names=feature_cols_label)
    data = wisconsin[feature_cols_label]
    array = data.values
    rawData = array[:,0:3]
    scaler = StandardScaler()
    rawData = scaler.fit_transform(rawData)

    split = train_test_split(rawData, y, test_size = 0.3, shuffle=True, random_state=randState)
    (train_raw_Data, test_raw_Data, train_raw_Target, test_raw_Target) = split


    col = ['component_1', 'component_2']
    PCA_data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/PCA/'
                           'pca_cancer_reduced.csv', names=col)
    PCA_data = PCA_data[col]
    PCA_data = PCA_data.values
    split = train_test_split(PCA_data, y, test_size = 0.3, shuffle=True, random_state=randState)
    (train_PCA_Data, test_PCA_Data, train_PCA_Target, test_PCA_Target) = split

    AE_data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/AutoEncoders/'
                          'AutoEncoder_Cancer_reduced.csv', names=col)
    AE_data = AE_data[col]
    AE_data = AE_data.values
    split = train_test_split(AE_data, y, test_size = 0.3, shuffle=True, random_state=randState)
    (train_AE_Data, test_AE_Data, train_AE_Target, test_AE_Target) = split

    RP_data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/'
                          'Randomized_Projection/Rand_Cancer_reduced.csv', names=col)
    RP_data = RP_data[col]
    RP_data= RP_data.values
    split = train_test_split(RP_data, y, test_size = 0.3, shuffle=True, random_state=randState)
    (train_RP_Data, test_RP_Data, train_RP_Target, test_RP_Target) = split

    train_raw_Accuracy = []
    test_raw_Accuracy = []
    train_PCA_Accuracy = []
    test_PCA_Accuracy = []
    train_AE_Accuracy = []
    test_AE_Accuracy = []
    train_RP_Accuracy = []
    test_RP_Accuracy = []


    x_values = np.arange(0.1, 1, 0.025)
    fig = 0
    for size in x_values:
        trainScore = []
        testScore = []
        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        for i in range(10):
            split = train_test_split(train_raw_Data, train_raw_Target, train_size=size, shuffle=True)
            (X_train, _, y_train, _) = split
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_train)
            trainScore.append(metrics.accuracy_score(y_train, y_pred))
            y_pred = mlp.predict(test_raw_Data)
            testScore.append(metrics.accuracy_score(test_raw_Target, y_pred))
        train_raw_Accuracy.append(np.mean(trainScore))
        test_raw_Accuracy.append(np.mean(testScore))
        # -----------------------------------------------------------------------------------------------------------------

        trainScore = []
        testScore = []
        mlp = MLPClassifier(hidden_layer_sizes=(2, 2, 2), max_iter=2000)
        for i in range(10):
            split = train_test_split(train_PCA_Data, train_PCA_Target, train_size=size, shuffle=True)
            (X_train, _, y_train, _) = split
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_train)
            trainScore.append(metrics.accuracy_score(y_train, y_pred))

            y_pred = mlp.predict(test_PCA_Data)
            testScore.append(metrics.accuracy_score(test_PCA_Target, y_pred))
        train_PCA_Accuracy.append(np.mean(trainScore))
        test_PCA_Accuracy.append(np.mean(testScore))
        # -----------------------------------------------------------------------------------------------------------------

        trainScore = []
        testScore = []
        mlp = MLPClassifier(hidden_layer_sizes=(2, 2, 2), max_iter=2000)
        for i in range(10):
            split = train_test_split(train_AE_Data, train_AE_Target, train_size=size, shuffle=True)
            (X_train, _, y_train, _) = split
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_train)
            trainScore.append(metrics.accuracy_score(y_train, y_pred))

            y_pred = mlp.predict(test_AE_Data)
            testScore.append(metrics.accuracy_score(test_AE_Target, y_pred))
        train_AE_Accuracy.append(np.mean(trainScore))
        test_AE_Accuracy.append(np.mean(testScore))
        # -----------------------------------------------------------------------------------------------------------------

        trainScore = []
        testScore = []
        mlp = MLPClassifier(hidden_layer_sizes=(2, 2, 2), max_iter=2000)
        for i in range(10):
            split = train_test_split(train_RP_Data, train_RP_Target, train_size=size, shuffle=True)
            (X_train, _, y_train, _) = split
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_train)
            trainScore.append(metrics.accuracy_score(y_train, y_pred))

            y_pred = mlp.predict(test_RP_Data)
            testScore.append(metrics.accuracy_score(test_RP_Target, y_pred))
        train_RP_Accuracy.append(np.mean(trainScore))
        test_RP_Accuracy.append(np.mean(testScore))
    # --------------------------------------------------------------------------------------------------------------------

    x_axes_values = [i*len(train_raw_Data) for i in x_values]

    fig = fig + 1
    plt.figure(fig)
    plt.plot(x_axes_values, train_raw_Accuracy, label='raw train')
    plt.plot(x_axes_values, train_PCA_Accuracy, label='PCA train')
    plt.plot(x_axes_values,train_AE_Accuracy, label='AE train')
    plt.plot(x_axes_values,train_RP_Accuracy, label='RP train')
    plt.title('Train Accuracy vs training size\nWisconsin Cancer Dataset')
    plt.xlabel('training size')
    plt.ylabel('train accuracy score')
    plt.legend(loc='best')
    plt.savefig('randtrainExp'+str(randState))

    fig = fig + 1
    plt.figure(fig)
    plt.plot(x_axes_values, test_raw_Accuracy, label='raw test')
    plt.plot(x_axes_values, test_PCA_Accuracy, label='PCA test')
    plt.plot(x_axes_values, test_AE_Accuracy, label='AE test')
    plt.plot(x_axes_values, test_RP_Accuracy, label='RP test')
    plt.title('Test Accuracy vs training size\nWisconsin Cancer Dataset')
    plt.xlabel('training size')
    plt.ylabel('test accuracy score')
    plt.legend(loc='best')
    plt.savefig('randtestExp'+str(randState))