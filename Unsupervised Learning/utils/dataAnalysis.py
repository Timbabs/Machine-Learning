import pandas as pd


from matplotlib import pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, names=col_names)

feature_cols = ['glucose', 'bmi', 'age']
data = pima[feature_cols]
X = data.ix[:,0:3].values
y = pima.ix[:,8].values


label_dict = ['diabetes_negative', 'diabetes_positive']

feature_dict = {0: 'glucose', 1: 'bp', 2: 'age'}

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for cnt in range(3):
        #plt.subplot(2, 2, cnt+1)
        for lab in [0, 1]:
            plt.figure(cnt)
            plt.hist(X[y==lab, cnt],
                     label=label_dict[lab],
                     bins=10,
                     alpha=0.3, edgecolor='black')
            plt.xlabel(feature_dict[cnt])
            plt.legend(loc='upper right', fancybox=True, fontsize=8)
            plt.savefig('diabetesDataAna' + str(cnt) + '.png')

    # plt.tight_layout()
    # plt.show()

# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
#
# from matplotlib import pyplot as plt
#
#
# col_names = ['worst_perimeter', 'mean_concativity', 'mean_area', 'label']
# wisconsin = pd.read_csv('data.csv', names=col_names)
# X = wisconsin.ix[:,0:3].values
# y = wisconsin.ix[:,3].values
#
#
# label_dict = ['WDBC-Malignant', 'WDBC-Benign']
#
# feature_dict = {0: 'worst_perimeter', 1: 'mean_concativity', 2: 'mean_area'}
#
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(8, 6))
#     for cnt in range(3):
#         #plt.subplot(2,2, cnt+1)
#         for lab in [0, 1]:
#             plt.figure(cnt)
#             plt.hist(X[y==lab, cnt],
#                      label=label_dict[lab],
#                      bins=10,
#                      alpha=0.3, edgecolor='black')
#             plt.xlabel(feature_dict[cnt])
#             plt.legend(loc='upper right', fancybox=True, fontsize=8)
#             plt.savefig('cancerDataAna' + str(cnt) + '.png')

    # plt.tight_layout()
    # plt.show()