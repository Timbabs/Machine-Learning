from sklearn.mixture import GMM
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics




url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, names=col_names)

feature_cols = ['glucose', 'bmi', 'age', 'label']
data = pima[feature_cols]
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
# ax2.axvline(x=3, ls='--',  c='black')
# ax1.text(3.1, 6250, 'n=3', verticalalignment='center')
#
#
#
# plt.title('Pima Indian Diabetes Dataset')
# plt.savefig('K-EM.png', dpi=300)
#
# plt.show()

#nitializing KMeans
gmm = GMM(n_components=3, random_state=0)
# Fitting with scaled inputs
gmm = gmm.fit(X)
labels = gmm.predict(X)


probs = gmm.predict_proba(X)


y_pred_prob = probs[:, 1]

# histogram of predicted probabilities
plt.rcParams['font.size'] = 14
fig1 = plt.figure(1)
plt.hist(probs[:,0], bins='auto', edgecolor='black', alpha=0.3, linewidth=1.2, label='cluster 1')
plt.hist(probs[:,2], bins='auto', edgecolor='black', alpha=0.3, linewidth=1.2, label='cluster 2')
plt.hist(probs[:,2], bins='auto', edgecolor='black', alpha=0.3, linewidth=1.2, label='cluster 3')
plt.axvline(x=0.5, ls='--',  c='black')
plt.legend(loc='best')
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probabilities')
plt.ylabel('Frequency')
fig1.savefig('2_custers_histEM.png', dpi=300)
plt.show()


# fig2 = plt.figure(2)
# ax = Axes3D(fig2)
#
# LABEL_COLOR_MAP = {0: 'g',
#                    1: 'r',
#                    2: 'purple',
#                    # 3: 'b',
#                    # 4: 'orange'
#                    }
#
# label_color = [LABEL_COLOR_MAP[l] for l in labels]
#
# #size = 50 * probs.max(1) ** 2  # square emphasizes differences
# ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=label_color)
#
# import matplotlib
#
# scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
# scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')
# scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[2], marker='o')
# ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['cluster 1', 'cluster 2', 'cluster 3'], numpoints=2)
#
#
# ax.set_xlabel('glucose')
# ax.set_ylabel('age')
# ax.set_zlabel('bmi')
# ax.set_title('3 clusters')
#
#
# #fig2.savefig('3_custers_EM.png', dpi=300)
#
#
# feature_cols_cp1 = ['glucose', 'bmi', 'age']
# feature_cols_cp2 = ['glucose', 'bmi', 'age']
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
#
# plt.show()


