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

feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
wisconsin = pd.read_csv('../utils/data.csv', names=feature_cols_label)
data = wisconsin[feature_cols_label]
array = data.values
rawData = array[:,0:3]
scaler = StandardScaler()
rawData = scaler.fit_transform(rawData)

EM_rawData = pd.DataFrame(rawData)
cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part1/Expectation__Maximization/'
                            'None+EM_clusters.csv', names=['val'])
EM_rawData['clusters'] = cluster_label
split = train_test_split(EM_rawData, y, test_size = 0.3, shuffle=True, random_state=2)
(train_EM_raw_Data, test_EM_raw_Data, train_EM_raw_Target, test_EM_raw_Target) = split

KM_rawData = pd.DataFrame(rawData)
cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part1/k-means_clustering/'
                            'None+KM_clusters.csv', names=['val'])
KM_rawData['clusters'] = cluster_label
split = train_test_split(KM_rawData, y, test_size = 0.3, shuffle=True, random_state=2)
(train_KM_raw_Data, test_KM_raw_Data, train_KM_raw_Target, test_KM_raw_Target) = split

col = ['component_1', 'component_2']
PCA_data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/PCA/'
                       'pca_cancer_reduced.csv', names=col)
cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part3/PCA_data/'
                                'cancer_data/PCA+EM_clusters.csv', names=['val'])
EM_PCA_data = PCA_data
EM_PCA_data['clusters'] = cluster_label
EM_PCA_data = EM_PCA_data.values
split = train_test_split(EM_PCA_data, y, test_size = 0.3, shuffle=True, random_state=2)
(train_EM_PCA_Data, test_EM_PCA_Data, train_EM_PCA_Target, test_EM_PCA_Target) = split

cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part3/PCA_data/'
                                'cancer_data/PCA+KM_clusters.csv', names=['val'])
KM_PCA_data = PCA_data
KM_PCA_data['clusters'] = cluster_label
KM_PCA_data = KM_PCA_data.values
split = train_test_split(KM_PCA_data, y, test_size = 0.3, shuffle=True, random_state=2)
(train_KM_PCA_Data, test_KM_PCA_Data, train_KM_PCA_Target, test_KM_PCA_Target) = split

AE_data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/AutoEncoders/'
                      'AutoEncoder_Cancer_reduced.csv', names=col)
cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part3/AutoEncoder_data/'
                            'cancer_data/AE+EM_clusters.csv', names=['val'])
EM_AE_data = AE_data
EM_AE_data['clusters'] = cluster_label
EM_AE_data = EM_AE_data.values
split = train_test_split(EM_AE_data, y, test_size = 0.3, shuffle=True, random_state=2)
(train_EM_AE_Data, test_EM_AE_Data, train_EM_AE_Target, test_EM_AE_Target) = split

cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part3/AutoEncoder_data/'
                            'cancer_data/AE+KM_clusters.csv', names=['val'])
KM_AE_data = AE_data
KM_AE_data['clusters'] = cluster_label
KM_AE_data = KM_AE_data.values
split = train_test_split(KM_AE_data, y, test_size = 0.3, shuffle=True, random_state=2)
(train_KM_AE_Data, test_KM_AE_Data, train_KM_AE_Target, test_KM_AE_Target) = split


RP_data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/'
                      'Randomized_Projection/Rand_Cancer_reduced.csv', names=col)
cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part3/RandProj_data/'
                            'cancer_data/RP+EM_clusters.csv', names=['val'])
EM_RP_data = RP_data
EM_RP_data['clusters'] = cluster_label
EM_RP_data = EM_RP_data.values
split = train_test_split(EM_RP_data, y, test_size = 0.3, shuffle=True, random_state=2)
(train_EM_RP_Data, test_EM_RP_Data, train_EM_RP_Target, test_EM_RP_Target) = split

cluster_label = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part3/RandProj_data/'
                            'cancer_data/RP+KM_clusters.csv', names=['val'])
KM_RP_data = RP_data
KM_RP_data['clusters'] = cluster_label
KM_RP_data = KM_RP_data.values
split = train_test_split(KM_RP_data, y, test_size = 0.3, shuffle=True, random_state=2)
(train_KM_RP_Data, test_KM_RP_Data, train_KM_RP_Target, test_KM_RP_Target) = split

train_EM_raw_Accuracy = []
train_KM_raw_Accuracy = []
test_EM_raw_Accuracy = []
test_KM_raw_Accuracy = []
train_EM_PCA_Accuracy = []
train_KM_PCA_Accuracy = []
test_EM_PCA_Accuracy = []
test_KM_PCA_Accuracy = []
train_EM_AE_Accuracy = []
train_KM_AE_Accuracy = []
test_EM_AE_Accuracy = []
test_KM_AE_Accuracy = []
train_EM_RP_Accuracy = []
train_KM_RP_Accuracy = []
test_EM_RP_Accuracy = []
test_KM_RP_Accuracy = []


x_values = np.arange(0.1, 1, 0.025)
No_iterations = 10

fig = 0
for size in x_values:
    train_EM_Score = []
    train_KM_Score = []
    test_EM_Score = []
    test_KM_Score = []

    for i in range(No_iterations):
        mlp = MLPClassifier(hidden_layer_sizes=(4, 4, 4), max_iter=2000)
        split = train_test_split(train_EM_raw_Data, train_EM_raw_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_EM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_EM_raw_Data)
        test_EM_Score.append(metrics.accuracy_score(test_EM_raw_Target, y_pred))

        mlp = MLPClassifier(hidden_layer_sizes=(4, 4, 4), max_iter=2000)
        split = train_test_split(train_KM_raw_Data, train_KM_raw_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_KM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_KM_raw_Data)
        test_KM_Score.append(metrics.accuracy_score(test_KM_raw_Target, y_pred))

    train_EM_raw_Accuracy.append(np.mean(train_EM_Score))
    test_EM_raw_Accuracy.append(np.mean(test_EM_Score))
    train_KM_raw_Accuracy.append(np.mean(train_KM_Score))
    test_KM_raw_Accuracy.append(np.mean(test_KM_Score))
#     # -----------------------------------------------------------------------------------------------------------------

    train_EM_Score = []
    train_KM_Score = []
    test_EM_Score = []
    test_KM_Score = []

    for i in range(No_iterations):
        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        split = train_test_split(train_EM_PCA_Data, train_EM_PCA_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_EM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_EM_PCA_Data)
        test_EM_Score.append(metrics.accuracy_score(test_EM_PCA_Target, y_pred))

        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        split = train_test_split(train_KM_PCA_Data, train_KM_PCA_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_KM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_KM_PCA_Data)
        test_KM_Score.append(metrics.accuracy_score(test_KM_PCA_Target, y_pred))

    train_EM_PCA_Accuracy.append(np.mean(train_EM_Score))
    test_EM_PCA_Accuracy.append(np.mean(test_EM_Score))
    train_KM_PCA_Accuracy.append(np.mean(train_KM_Score))
    test_KM_PCA_Accuracy.append(np.mean(test_KM_Score))
#     # -----------------------------------------------------------------------------------------------------------------
#
    train_EM_Score = []
    train_KM_Score = []
    test_EM_Score = []
    test_KM_Score = []

    for i in range(No_iterations):
        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        split = train_test_split(train_EM_AE_Data, train_EM_AE_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_EM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_EM_AE_Data)
        test_EM_Score.append(metrics.accuracy_score(test_EM_AE_Target, y_pred))

        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        split = train_test_split(train_KM_AE_Data, train_KM_AE_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_KM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_KM_AE_Data)
        test_KM_Score.append(metrics.accuracy_score(test_KM_AE_Target, y_pred))

    train_EM_AE_Accuracy.append(np.mean(train_EM_Score))
    test_EM_AE_Accuracy.append(np.mean(test_EM_Score))
    train_KM_AE_Accuracy.append(np.mean(train_KM_Score))
    test_KM_AE_Accuracy.append(np.mean(test_KM_Score))
#     # -----------------------------------------------------------------------------------------------------------------
#
    train_EM_Score = []
    train_KM_Score = []
    test_EM_Score = []
    test_KM_Score = []

    for i in range(No_iterations):
        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        split = train_test_split(train_EM_RP_Data, train_EM_RP_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_EM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_EM_RP_Data)
        test_EM_Score.append(metrics.accuracy_score(test_EM_RP_Target, y_pred))

        mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=2000)
        split = train_test_split(train_KM_RP_Data, train_KM_RP_Target, train_size=size, shuffle=True)
        (X_train, _, y_train, _) = split
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        train_KM_Score.append(metrics.accuracy_score(y_train, y_pred))
        y_pred = mlp.predict(test_KM_RP_Data)
        test_KM_Score.append(metrics.accuracy_score(test_KM_RP_Target, y_pred))

    train_EM_RP_Accuracy.append(np.mean(train_EM_Score))
    test_EM_RP_Accuracy.append(np.mean(test_EM_Score))
    train_KM_RP_Accuracy.append(np.mean(train_KM_Score))
    test_KM_RP_Accuracy.append(np.mean(test_KM_Score))
# # --------------------------------------------------------------------------------------------------------------------
#
x_axes_values = [i*len(rawData) for i in x_values]

fig = fig + 1
plt.figure(fig)
plt.plot(x_axes_values, train_EM_raw_Accuracy, label='raw+EM train')
plt.plot(x_axes_values, train_KM_raw_Accuracy, label='raw+KM train')
plt.plot(x_axes_values, train_EM_PCA_Accuracy, label='PCA+EM train')
plt.plot(x_axes_values, train_KM_PCA_Accuracy, label='PCA+KM train')
plt.plot(x_axes_values,train_EM_AE_Accuracy, label='AE+EM train')
plt.plot(x_axes_values,train_KM_AE_Accuracy, label='AE+KM train')
plt.plot(x_axes_values,train_EM_RP_Accuracy, label='RP+EM train')
plt.plot(x_axes_values,train_KM_RP_Accuracy, label='RP+KM train')
plt.title('Train Accuracy vs training size\nWisconsin Cancer Dataset')
plt.xlabel('training size')
plt.ylabel('train accuracy score')
plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
plt.savefig('train')

fig = fig + 1
plt.figure(fig)
plt.plot(x_axes_values, test_EM_raw_Accuracy, label='raw+EM test')
plt.plot(x_axes_values, test_KM_raw_Accuracy, label='raw+KM test')
plt.plot(x_axes_values, test_EM_PCA_Accuracy, label='PCA+EM test')
plt.plot(x_axes_values, test_KM_PCA_Accuracy, label='PCA+KM test')
plt.plot(x_axes_values,test_EM_AE_Accuracy, label='AE+EM test')
plt.plot(x_axes_values,test_KM_AE_Accuracy, label='AE+KM test')
plt.plot(x_axes_values,test_EM_RP_Accuracy, label='RP+EM test')
plt.plot(x_axes_values,test_KM_RP_Accuracy, label='RP+KM test')
plt.title('Test Accuracy vs training size\nWisconsin Cancer Dataset')
plt.xlabel('training size')
plt.ylabel('test accuracy score')
plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
plt.savefig('test')

plt.show()
