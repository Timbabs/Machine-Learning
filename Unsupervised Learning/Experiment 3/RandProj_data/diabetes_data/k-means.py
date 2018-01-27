import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

# --------------------------------------------------------------------------------------------------------------------

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, names=col_names)

y = pima['label']

col = ['component_1', 'component_2']
data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/part2/Randomized_Projection'
                   '/Rand_diabetes_reduced.csv', names=col)

data = data[col]
X = data.values



# k_range = list(range(1))
# k_range = range(1, 40)
# cluster_errors = []
# ARI = []
# for k in k_range:
#     k_means = KMeans(n_clusters=k, random_state=0)
#     k_means.fit(X)
#     cluster_errors.append(k_means.inertia_)
#
#     labels = k_means.labels_
#     ARI.append(metrics.adjusted_rand_score(y, labels))
#
#
# fig, ax1 = plt.subplots()
# plt1 = ax1.plot(k_range, cluster_errors, 'b')
# ax1.set_ylabel('within-cluster sum of squares', color='b')
# ax1.tick_params('y', colors='b')
# ax1.set_xlabel("k (Number of clusters)")
#
#
# ax2 = ax1.twinx()
# ax2.plot(k_range, ARI, 'r')
# ax2.set_ylabel('Adjusted rand Index', color='r')
# ax2.tick_params('y', colors='r')
# ax2.axvline(x=2, ls='--',  c='black')
# ax1.text(2.5, 500, 'k=2', verticalalignment='center')
# ax2.axvline(x=6, ls='--',  c='black')
# ax1.axhline(y = 120, ls='--',  c='black')
# ax1.text(6.5, 1500, 'k=10', verticalalignment='center')
#
#
# plt.title('Pima Indian Diabetes RP-reduced dataset')
# plt.savefig('updateDiabetes.png', dpi=300)
#
# plt.show()

#
# Initializing KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
# Fitting with scaled inputs
kmeans = kmeans.fit(X)

labels = kmeans.labels_

# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure(1)

LABEL_COLOR_MAP = {0: 'g', 1: 'r', 2: 'b', 3: 'orange', 4: 'y',  5: 'purple', 6: 'k',  7: 'brown',  8: 'm',  9: 'c'}

label_color = [LABEL_COLOR_MAP[l] for l in labels]

plt.scatter(X[:, 0], X[:, 1], c=label_color)
# ax.legend()
import matplotlib

scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')

#
plt.legend([scatter1_proxy, scatter2_proxy],
          ['cluster 1', 'cluster 2'], numpoints=2, loc='best')

plt.xlabel('component1')
plt.ylabel('component2')


plt.plot(C[:, 0], C[:, 1], marker='*', c='black', ms=15)

plt.savefig('diabetes2.png', dpi=300)
plt.title('Pima Diabetes RP-reduced dataset')


feature_cols_cp1 = ['component_1', 'component_2']
feature_cols_cp2 = ['component_1', 'component_2']

#Glue back to originaal data
data['clusters'] = labels

#Add the column into our list
feature_cols_cp1.extend(['clusters'])
#Lets analyze the clusters
print (data[feature_cols_cp1].groupby(['clusters']).mean())
print('\n')
#Glue back to originaal data
data['trueClass'] = y

#Add the column into our list
feature_cols_cp2.extend(['trueClass'])
#Lets analyze the clusters
print (data[feature_cols_cp2].groupby(['trueClass']).mean())

similarity = metrics.adjusted_rand_score(y, labels)
print("\nsimilarity: " + str(similarity))

plt.show()
