
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import pylab


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, names=col_names)

feature_cols = ['glucose', 'bmi', 'age', 'label']

array = pima[feature_cols].values
X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]

# -----------------------------------------------------------------------------------------------------------------

sp = GaussianRandomProjection(n_components=2, random_state=0)
fit = sp.fit(X)



fig1 = plt.figure(1)
ax = Axes3D(fig1)

X_sp = sp.transform(X)

# df = pd.DataFrame(X_sp)
# df.to_csv('Rand_diabetes_reduced.csv', index=0, header=0)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.2, label='original data (3 features)')
ax.scatter(X_sp[:, 0], X_sp[:, 1], alpha=0.8, label='projected data (2 components)')
ax.legend(loc='best')
ax.set_title('Randomized \nProjection')

plt.savefig('RandcomboProj.png', dpi=300)
pylab.show()

# projected = sp.transform(X)
# plt.scatter(projected[:, 0], projected[:, 1],
#             c=y, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('spectral', 2))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.savefig('RandComb4.png', dpi=300)
# plt.show()
