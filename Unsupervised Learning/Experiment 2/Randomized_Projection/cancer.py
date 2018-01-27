
# from sklearn.random_projection import johnson_lindenstrauss_min_dim
# print(johnson_lindenstrauss_min_dim(768,eps=0.1))

# ----------------------------------------------------------------------------------------------------------------------

from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
wisconsin = pd.read_csv('../data.csv', names=feature_cols_label)

array = wisconsin[feature_cols_label].values
X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]

# -----------------------------------------------------------------------------------------------------------------

sp = GaussianRandomProjection(n_components=2, random_state=18)
fit = sp.fit(X)
fig1 = plt.figure(1)




ax = Axes3D(fig1)

X_sp = sp.transform(X)

df = pd.DataFrame(X_sp)
df.to_csv('Rand_Cancer_reduced.csv', index=0, header=0)
#
#
ax.scatter(X[:, 1], X[:, 0], X[:, 2], alpha=0.2, label='original data (3 features)')
ax.scatter(X_sp[:, 1], X_sp[:, 0], alpha=0.8, label='projected data (2 components)')
ax.legend(loc='best')
ax.set_title('Randomized \nProjection')

plt.savefig('RandcomboProjCancer.png', dpi=300)
pylab.show()
#
#
# projected = sp.transform(X)
# plt.scatter(projected[:, 0], projected[:, 1],
#             c=y, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('spectral', 2))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.savefig('RandCombCancer.png', dpi=300)
# plt.show()
