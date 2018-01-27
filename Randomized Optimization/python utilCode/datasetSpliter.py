from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sk
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()

data = np.ndarray(shape=(569, 31), dtype=float, order='F')
for i in range(0, 30):
    data[:, i] = cancer.data[:, i]
data = sk.normalize(data)
data[:, 30] = cancer.target

trainData, testData = train_test_split(data, shuffle=True, train_size=0.7, random_state=6)

df_train = pd.DataFrame(trainData)
df_test = pd.DataFrame(testData)
df_train.to_csv("../Assignment2/ABAGAIL/src/opt/test/myTests/inputData/train_70.csv", index=0, header=0)
df_test.to_csv("../Assignment2/ABAGAIL/src/opt/test/myTests/inputData/test_30.csv", index=0, header=0)

splits = [0.1, 0.2, 0.3, 0.4, 0.5]
for split in splits:
    split1, split2 = train_test_split(trainData, shuffle=True, train_size=split)
    df_split1 = pd.DataFrame(split1)
    df_split2 = pd.DataFrame(split2)
    df_split1.to_csv("../Assignment2/ABAGAIL/src/opt/test/myTests/inputData/train_70_" + str(int(split*100))+".csv", index=0, header=0)
    df_split2.to_csv("../Assignment2/ABAGAIL/src/opt/test/myTests/inputData/train_70_" + str(int((1-split)*100))+".csv", index=0, header=0)

