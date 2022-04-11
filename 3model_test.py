from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#1读入数据
gd2 = pd.read_excel('d00_3.xls',header = None,index_col = None) #cell8 4个特征+capa
x_train = gd2.iloc[0:,0:4]
y_train = gd2.iloc[0:,4]

#2建立MLR模型
clf = linear_model.LinearRegression()
clf.fit(x_train, y_train)
y_train1 = clf.predict(x_train)  
df1 = pd.DataFrame(y_train1)
df1.to_excel('y_train1.xlsx')

#3建立SVR模型
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
svr.fit(x_train,y_train)
y_train2 = svr.predict(x_train)
df2 = pd.DataFrame(y_train2)
df2.to_excel('y_train2.xlsx')

#4建立GPR模型
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(x_train,y_train)
y_train3 = gpr.predict(x_train)
df3 = pd.DataFrame(y_train3)
df3.to_excel('y_train3.xlsx')

