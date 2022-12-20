import pandas as pd
import numpy as np

df = pd.read_csv("bostonhousing.csv")
x = df.drop(columns=['medv'])
y = df[['medv']]
df.describe()

df.info()
colname= [i for i in (df.columns.drop('medv'))]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train.info()


from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))



################################################
model_xg = XGBRegressor()
model_xg.fit(X_train, y_train, verbose=False)
pred_xg = model_xg.predict(X_test)
model_xg.score(X_test,y_test) #모델 설명력:R2
print("XGB Regressor 평가지표")
print("R2_score : ", r2_score(y_test,pred_xg))
print("RMSE : ", str(rmse(y_test, pred_xg)))
###################################################
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
pred_rf= model_rf.predict(X_test)
model_rf.score(X_test,y_test) #모델 설명력 :R2
print("RandomForest Regressor 평가지표" )
print("R2_score : " , r2_score(y_test,pred_rf))
print("RMSE : " , str(rmse(y_test, pred_rf)))
###################################################
#중요변수 시각화
ftr_importances_values = model_rf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=colname)
ftr_top = ftr_importances.sort_values(ascending=False)[:20]

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(x=ftr_top, y=ftr_top.index)
plt.title("Importance Value")
plt.show()


###회귀 시각화

import matplotlib.pyplot as plt

x = y_test['medv']
y2=pred_rf
plt.figure(figsize = (8,6))
plt.title("RandomForestRegressor Model")
plt.plot(x, y2, 'o', color = 'r')
m, b = np.polyfit(x, y2, 1)
plt.xlabel('Condition')
plt.ylabel('Prediction')
plt.plot(x, m * x + b, color = 'darkblue')

x = y_test['medv']
y = pred_xg
plt.figure(figsize = (8,6))
plt.title("XGBRegressor Model")
plt.plot(x, y, 'o', color = 'r')
m, b = np.polyfit(x, y, 1)
plt.xlabel('Condition')
plt.ylabel('Prediction')
plt.plot(x, m * x + b, color = 'darkblue')


plt.figure(figsize = (8,6))
sns.distplot(df['medv'], color = 'darkorange')
plt.title("Distribution of medv")
plt.show()

