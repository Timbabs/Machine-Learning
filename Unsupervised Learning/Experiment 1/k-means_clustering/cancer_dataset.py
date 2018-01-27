from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

cancer = load_breast_cancer()


#Feature selection
# X = cancer.data
#
# Y = cancer.target
# # feature extraction
# model = ExtraTreesClassifier(random_state=0)
# model.fit(X, Y)
# #print(model.feature_importances_)
# result = [i for i in model.feature_importances_]
#
# dic = {counter:  value for (counter, value) in enumerate(result)}
#
#     #print(counter, value)
#
# ans = []
# for _ in range(0, 3):
#     max = 0.0
#     for i in result:
#         if i > max:
#             max = i
#     ans.append(max)
#     result.remove(max)
# #print(ans)
#
# for j in ans:
#     for i in range(0, len(dic)):
#         if dic[i] is j:
#             print(i)

#Data to csv
# data = np.ndarray(shape=(569, 4), dtype=float, order='F')
# data[:, 0] = cancer.data[:, 22]
# data[:, 1] = cancer.data[:, 6]
# data[:, 2] = cancer.data[:, 3]
# data[:, 3] = cancer.target
#
# df_split1 = pd.DataFrame(data)
# df_split1.to_csv("data.csv", header=0, index=0,)
# #

feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
wisconsin = pd.read_csv('data.csv', names=feature_cols_label)

data = wisconsin[feature_cols_label]
array = data.values

X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]


#
#search for an optimal value of K for K_means
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
# plt1 = ax1.plot(k_range, cluster_errors, 'b-')
# ax1.set_ylabel('within-cluster sum of squares', color='b')
# ax1.tick_params('y', colors='b')
# ax1.set_xlabel("k (Number of clusters)")
#
#
#
#
# ax2 = ax1.twinx()
# ax2.plot(k_range, ARI, 'r-')
# ax2.set_ylabel('Adjusted rand Index', color='r')
# ax2.tick_params('y', colors='r')
# ax2.axvline(x=2, ls='--',  c='black')
# ax2.axvline(x=6, ls='--', c='black')
# ax1.text(2.1, 100, 'k=2', verticalalignment='center')
# ax1.text(6.1, 500, 'k=6', verticalalignment='center')
#
# plt.title('Wisconsin Cancer Dataset')
# plt.savefig('updated.png', dpi=300)



# Initializing KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
# Fitting with scaled inputs
kmeans = kmeans.fit(X)

labels = kmeans.labels_

df = pd.DataFrame(labels)
df.to_csv('None+KM_clusters.csv', index=0, header=0)

# # Getting the cluster centers
# C = kmeans.cluster_centers_
#
# fig = plt.figure(1)
# ax = Axes3D(fig)
#
# LABEL_COLOR_MAP = {0: 'g', 1: 'r', 2: 'b', 3: 'orange', 4: 'y',  5: 'purple'}
#
# label_color = [LABEL_COLOR_MAP[l] for l in labels]
#
# ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=label_color)
# ax.legend()
# import matplotlib
#
# scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
# scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')
# # scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[2], marker='o')
# # scatter4_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[3], marker='o')
# # scatter5_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[4], marker='o')
# # scatter6_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[5], marker='o')
#
#
# ax.legend([scatter1_proxy, scatter2_proxy, '''scatter3_proxy, scatter4_proxy, scatter5_proxy, scatter6_proxy'''],
#           ['cluster 1', 'cluster 2'], numpoints=2)
#
#
# ax.set_xlabel('worst_perimeter')
# ax.set_ylabel('mean_area')
# ax.set_zlabel('mean_concativity')
# ax.set_title('2 clusters')
#
# ax.plot(C[:, 0], C[:, 2], C[:, 1], marker='*', c='black', ms=15)
#
# plt.savefig('UPDATE2.png', dpi=300)
#
#
# feature_cols_cp1 = ['worst_perimeter', 'mean_concativity', 'mean_area']
# feature_cols_cp2 = ['worst_perimeter', 'mean_concativity', 'mean_area']
#
# #Glue back to originaal data
# data['clusters'] = labels
#
# #Add the column into our list
# feature_cols_cp1.extend(['clusters'])
# #Lets analyze the clusters
# print (data[feature_cols_cp1].groupby(['clusters']).mean())
# print('\n')
# #Glue back to originaal data
# data['trueClass'] = y
#
# #Add the column into our list
# feature_cols_cp2.extend(['trueClass'])
# #Lets analyze the clusters
# print (data[feature_cols_cp2].groupby(['trueClass']).mean())
#
# similarity = metrics.adjusted_rand_score(y, labels)
# print("\nsimilarity: " + str(similarity))
# #
# plt.show()
