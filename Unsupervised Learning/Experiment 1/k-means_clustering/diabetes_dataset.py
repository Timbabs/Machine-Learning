import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from matplotlib.pyplot import text

# --------------------------------------------------------------------------------------------------------------------

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, names=col_names)

# Feature selection
# from sklearn.ensemble import ExtraTreesClassifier
# array = pima.values
# X = array[:,0:8]
# Y = array[:,8]
# # feature extraction
# model = ExtraTreesClassifier()
# model.fit(X, Y)
#print(model.feature_importances_)

# k(number of clusters) selection
feature_cols = ['glucose', 'bmi', 'age', 'label']
data = pima[feature_cols]
array = data.values
X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]


#
# search for an optimal value of K for K_means
#k_range = list(range(1))
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
# ax2.axvline(x=10, ls='--',  c='black')
# ax1.text(10.5, 1500, 'k=10', verticalalignment='center')
#
#
# plt.title('Pima Indian Diabetes Dataset')
# plt.savefig('updateDiabetes.png', dpi=300)
#
# plt.show()

#
# Initializing KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
# Fitting with scaled inputs
kmeans = kmeans.fit(X)

labels = kmeans.labels_

# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure(1)
ax = Axes3D(fig)

LABEL_COLOR_MAP = {0: 'g', 1: 'r', 2: 'b', 3: 'orange', 4: 'y',  5: 'purple', 6: 'k',  7: 'brown',  8: 'm',  9: 'c'}

label_color = [LABEL_COLOR_MAP[l] for l in labels]

ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=label_color)
# ax.legend()
import matplotlib

scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')
scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[2], marker='o')
scatter4_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[3], marker='o')
scatter5_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[4], marker='o')
scatter6_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[5], marker='o')
scatter7_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[6], marker='o')
scatter8_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[7], marker='o')
scatter9_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[8], marker='o')
scatter10_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[9], marker='o')
#
ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy, scatter4_proxy, scatter5_proxy, scatter6_proxy,
           scatter7_proxy, scatter8_proxy, scatter9_proxy, scatter10_proxy],
          ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7'
           , 'cluster 8', 'cluster 9', 'cluster 10'], numpoints=2, loc='upper left')



ax.set_xlabel('glucose')
ax.set_ylabel('bmi')
ax.set_zlabel('age')
ax.set_title('10 clusters')

ax.plot(C[:, 0], C[:, 2], C[:, 1], marker='*', c='black', ms=15)

plt.savefig('UPDATEDiabetes10.png', dpi=300)


feature_cols_cp1 = ['glucose', 'bmi', 'age']
feature_cols_cp2 = ['glucose', 'bmi', 'age']

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
