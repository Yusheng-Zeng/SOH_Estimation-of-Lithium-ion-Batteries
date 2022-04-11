from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#1读入数据
#https://blog.csdn.net/fgg1234567890/article/details/110295368
gd1 = pd.read_excel('d00_2.xls',header = None,index_col = None)#3列cell1-7预测值+capa
x_train = gd1.iloc[0:,0:3] #左闭右开
y_train = gd1.iloc[0:,3] #capa放在第四列

te = pd.read_excel('d00_4.xls',header = None,index_col = None)#预测数据
x_test = te.iloc[0:,0:3]
y_test = te.iloc[0:,3]

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(x_train, y_train)
y_test1 = forest.predict(x_test)
df = pd.DataFrame(y_test1)
df.to_excel('y_test1.xlsx')