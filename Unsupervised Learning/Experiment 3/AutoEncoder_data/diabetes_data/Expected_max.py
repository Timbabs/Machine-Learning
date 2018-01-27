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

y = pima['label']

col = ['component_1', 'component_2']
data = pd.read_csv('~/Documents/School_Work/Fall_2017/CS4641/Assignment3/AutoEncoders/AutoEncoder_Diabetes_reduced.csv', names=col)

data = data[col]
X = data.values



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
# ax1.text(2.1, -6600, 'n=2', verticalalignment='center')
#
#
#
# plt.title('Pima Diabetes AutoEncoders-reduced dataset')
# plt.savefig('K-EM.png', dpi=300)
#
# plt.show()

# #nitializing KMeans
gmm = GMM(n_components=2, random_state=0)
# Fitting with scaled inputs
gmm = gmm.fit(X)
labels = gmm.predict(X)


probs = gmm.predict_proba(X)


#y_pred_prob = probs[:, 1]

#histogram of predicted probabilities
plt.rcParams['font.size'] = 14
fig1 = plt.figure(1)
plt.hist(probs[:, 0], bins='auto', edgecolor='black', linewidth=1.2, alpha=0.3, label='cluster 1')
plt.hist(probs[:, 1], bins='auto', edgecolor='black', linewidth=1.2,  alpha=0.3, label='cluster 2')
#plt.axvline(x=0.5, ls='--',  c='black')
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probabilities')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.show()
fig1.savefig('2_custers_histEM.png', dpi=300)


fig2 = plt.figure(2)

LABEL_COLOR_MAP = {0: 'g',
                   1: 'r',
                   # 2: 'purple',
                   # 3: 'b',
                   # 4: 'orange'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in labels]

#size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=label_color)

import matplotlib

scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')
plt.legend([scatter1_proxy, scatter2_proxy], ['cluster 1', 'cluster 2'], numpoints=2)


plt.xlabel('component1')
plt.ylabel('component2')

plt.title('EM on Pima Diabetes AutoEncoders-reduced data')
fig2.savefig('2_custers_EM.png', dpi=300)



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
#
#
