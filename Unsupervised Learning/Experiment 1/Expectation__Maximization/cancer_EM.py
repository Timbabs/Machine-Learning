from sklearn.datasets import load_breast_cancer
from sklearn.mixture import GMM
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

cancer = load_breast_cancer()


feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
wisconsin = pd.read_csv('data.csv', names=feature_cols_label)

data = wisconsin[feature_cols_label]
array = data.values

X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]




#search for an optimal value of K for K_means
# k_range = range(1, 40)
# bic = []
# ARI = []
# for k in k_range:
#     gmm = GMM(n_components=k, covariance_type='full')
#     gmm.fit(X)
#     labels = gmm.predict(X)
#     bic.append(gmm.bic(X))
#     ARI.append(metrics.adjusted_rand_score(y, labels))
#
#
#
# fig, ax1 = plt.subplots()
# ax1.plot(k_range, bic, 'b', label='BIC')
# ax1.set_ylabel('BIC Information criterion')
# ax1.tick_params('y', colors='b')
# ax1.set_xlabel("n (Number of clusters)")
#
#
# ax2 = ax1.twinx()
# ax2.plot(k_range, ARI, 'r')
# ax2.set_ylabel('Adjusted rand Index', color='r')
# ax2.tick_params('y', colors='r')
# ax2.axvline(x=2, ls='--',  c='black')
# ax1.text(2.1, 2600, 'n=2', verticalalignment='center')
#
#
#
# plt.title('Wisconsin Cancer Dataset')
# plt.savefig('k-Cancer_EM.png', dpi=300)
#
# plt.show()



# # Initializing KMeans
gmm = GMM(n_components=2, random_state=5)
# Fitting with scaled inputs
gmm = gmm.fit(X)
labels = gmm.predict(X)

# df = pd.DataFrame(labels)
# df.to_csv('None+EM_clusters.csv', index=0, header=0)

# probs = gmm.predict_proba(X)
#
#
# y_pred_prob = probs[:, 0]
#
# # histogram of predicted probabilities
# plt.rcParams['font.size'] = 14
# fig1 = plt.figure(1)
# plt.hist(probs[:,0], bins='auto', edgecolor='black', alpha=0.3, linewidth=1.2, label='cluster 1')
# plt.hist(probs[:,1], bins='auto', edgecolor='black', alpha=0.3, linewidth=1.2, label='cluster 2')
# plt.axvline(x=0.5, ls='--',  c='black')
# plt.legend(loc='center')
# plt.xlim(0, 1)
# plt.title('Histogram of predicted probabilities')
# plt.xlabel('Predicted probabilities')
# plt.ylabel('Frequency')
# fig1.savefig('2_clusterCancer_histEM.png', dpi=300)
# plt.show()
fig = plt.figure(1)
ax = Axes3D(fig)

LABEL_COLOR_MAP = {0: 'g',
                   1: 'r'}

label_color = [LABEL_COLOR_MAP[l] for l in labels]

ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=label_color)

import matplotlib

scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')
ax.legend([scatter1_proxy, scatter2_proxy], ['cluster 1', 'cluster 2'], numpoints=2)


ax.set_xlabel('worst_perimeter')
ax.set_ylabel('mean_area')
ax.set_zlabel('mean_concativity')
ax.set_title('2 clusters')

#plt.savefig('2_custersCancer_EM.png', dpi=300)


feature_cols_cp1 = ['worst_perimeter', 'mean_concativity', 'mean_area']
feature_cols_cp2 = ['worst_perimeter', 'mean_concativity', 'mean_area']

#Glue back to originaal data
data['clusters'] = labels

#Add the column into our list
feature_cols_cp1.extend(['clusters'])
#Lets analyze the clusters
print (data[feature_cols_cp1].groupby(['clusters']).mean())

#Glue back to originaal data
data['trueClass'] = y

#Add the column into our list
feature_cols_cp2.extend(['trueClass'])
#Lets analyze the clusters
print (data[feature_cols_cp2].groupby(['trueClass']).mean())

similarity = metrics.adjusted_rand_score(y, labels)
print("\nsimilarity: " + str(similarity))

plt.show()


