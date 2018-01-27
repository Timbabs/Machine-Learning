
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sknn import ae, mlp

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, names=col_names)

feature_cols = ['glucose', 'bmi', 'age', 'label']

array = pima[col_names].values
X = array[:,0:8]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,8]

# Initialize auto-encoder for unsupervised learning.


AELayers = [ae.Layer("Sigmoid", units=8), ae.Layer("Sigmoid", units=5), ae.Layer("Sigmoid", units=2)]

myae = ae.AutoEncoder(layers=AELayers,  random_state=0, learning_rate=0.002)

myae.fit(X)


fig1 = plt.figure(1)
ax = Axes3D(fig1)
# #
# X_sp = myae.transform(X)
#
#
# #ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.2, label='original data (3 features)')
# ax.scatter(X_sp[:, 0], X_sp[:, 1], alpha=0.8, label='projected data (2 components)')
# ax.legend(loc='best')
# ax.set_title('Randomized \nProjection')
# #
# # plt.savefig('RandcomboProj.png', dpi=300)
# plt.show()

projected = myae.transform(X)
df = pd.DataFrame(projected)
df.to_csv('AutoEncoder_Diabetes_reduced.csv', index=0, header=0)
ax.scatter(projected[:, 0], projected[:, 1],
            c=y)
# ax.set_xlabel('component 1')
# ax.set_ylabel('component 2')

ax.set_title('AutoEncoders')

#plt.savefig('3_AutoEncoders.png', dpi=300)
plt.show()




