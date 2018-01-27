
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from mpl_toolkits.mplot3d import Axes3D

feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
wisconsin = pd.read_csv('../../utils/data.csv', names=feature_cols_label)

array = wisconsin[feature_cols_label].values
X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]


# -----------------------------------------------------------------------------------------------------------------

pca = PCA(n_components=2)
fit = pca.fit(X)


# fig1 = plt.figure(1)
# ax = Axes3D(fig1)
# #
# ax.scatter(X[:, 0], X[:, 2], X[:, 1], alpha=0.2)
#
# x22, y22, _ = proj3d.proj_transform((pca.mean_)[0], (pca.mean_)[2], (pca.mean_)[1], ax.get_proj())
# v0 = (x22, y22)
#
# def draw_vector(v):
#     arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
#     pylab.annotate('', xy = v, xytext=v0, arrowprops = arrowprops)
#
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     x2, y2, _ = proj3d.proj_transform((pca.mean_ + v)[0], (pca.mean_ + v)[2], (pca.mean_ + v)[1], ax.get_proj())
#     v = (x2, y2)
#     draw_vector(v)
#
# ax.set_xlabel('worst_perimeter')
# ax.set_ylabel('mean_area')
# ax.set_zlabel('mean_concativity')
# ax.set_title('input features')
# plt.savefig('inputProjCancer.png', dpi=300)
# pylab.show()




# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops = dict(arrowstyle='->',
#                       linewidth=2,
#                       shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)
#
# # plot data
#
# fig = plt.figure(2)
# X = pca.transform(X)
# pca.fit(X)
#
#
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
#
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.title('principal components')
# plt.savefig('principalProjCancer.png', dpi=300)
# plt.show()



#
# # #
# fig1 = plt.figure(1)
# ax = Axes3D(fig1)
#
# X_pca = pca.transform(X)
#
# # df = pd.DataFrame(X_pca)
# # df.to_csv('pca_cancer_reduced.csv', index=0, header=0)
#
# X_new = pca.inverse_transform(X_pca)
#
# ax.scatter(X[:, 2], X[:, 1], X[:, 0], alpha=0.2, label='original data (3 features)')
# ax.scatter(X_new[:, 2], X_new[:, 1], X[:, 0], alpha=0.8, label='projected data (2 components)')
# ax.legend(loc='best')
#
# #plt.savefig('comboProjCancer.png', dpi=300)
# pylab.show()


# projected = pca.transform(X)
# plt.scatter(projected[:, 0], projected[:, 1],
#             c=y, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('spectral', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.savefig('3_PCACancer.png', dpi=300)
# plt.show()


# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
cov_mat = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print(eig_vals)
#
# tot = sum(eig_vals)
# var_exp = [(i / tot) for i in sorted(eig_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# plt.bar(range(1, 4), var_exp, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1, 4), cum_var_exp, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.savefig('k_cancerPCA.png')
# plt.show()

# -----------------------------------------------------------------------------------------------------------------
#
# fig2 = plt.figure(2)
# ax = Axes3D(fig2)
#
# LABEL_COLOR_MAP = {0: 'g',
#                    1: 'r'
#                    }
# #
# label_color = [LABEL_COLOR_MAP[l] for l in y]
#
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label_color)
# #
# #
# import matplotlib
#
# scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[0], marker='o')
# scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=LABEL_COLOR_MAP[1], marker='o')
# ax.legend([scatter1_proxy, scatter2_proxy], ['diabetes negative', 'diabetes positive'], numpoints=2)
#
#
# ax.set_xlabel('principal component 1')
# ax.set_ylabel('principal component 2')
# ax.set_zlabel('principal component 3')
# ax.set_title('3 Principal components Analysis')
#
#
# fig2.savefig('3_PCA.png', dpi=300)
#
#
# plt.show()
#
