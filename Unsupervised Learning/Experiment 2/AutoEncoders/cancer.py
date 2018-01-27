
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
from sklearn.datasets import load_breast_cancer


feature_cols_label = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
wisconsin = pd.read_csv('../data.csv', names=feature_cols_label)

array = wisconsin[feature_cols_label].values
X = array[:,0:3]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = array[:,3]

# Initialize auto-encoder for unsupervised learning.


AELayers = [ae.Layer("Sigmoid", units=3), ae.Layer("Sigmoid", units=3), ae.Layer("Sigmoid", units=2)]

myae = ae.AutoEncoder(layers=AELayers,  learning_rate=0.002)

myae.fit(X)



fig2 = plt.figure(2)
ax = Axes3D(fig2)

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
# ax.legend([scatter1_proxy, scatter2_proxy], ['WDBC-Malignant', 'WDBC-Benign'], numpoints=2)
#
#
# # ax.set_xlabel('component 1')
# # ax.set_ylabel('component 2')
# # ax.set_zlabel('component 3')
# ax.set_title('AutoEncoders')
# fig2.savefig('3_AutoEncodersCancer.png', dpi=300)
#
#
# plt.show()
#

projected = myae.transform(X)
df = pd.DataFrame(projected)
df.to_csv('AutoEncoder_Cancer_reduced.csv', index=0, header=0)
ax.scatter(projected[:, 0], projected[:, 1],
            c=y)

ax.set_title('AutoEncoders')

plt.savefig('3_AutoEncodersCancer.png', dpi=300)
plt.show()
