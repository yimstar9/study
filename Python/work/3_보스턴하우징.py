from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #(4)sklearn.tree패키지에 DecisionTreeClassifier import
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import xgboost
#from xgboost import DMatrix
import pandas as pd
import numpy as np
import seaborn as sns
import math
#다만, 분류에서는 model.predict_proba( )로 뽑아서 roc-auc로 평가하고
# 회귀에서는 model.predict로 뽑아서 r2 score나 MSE등의 지표로 평가한다는 게 다를 것 같네요.

#(2)데이터 준비
df = pd.read_csv("BostonHousing.csv")
#데이터 분리
df
x = df.drop(columns=['medv'])
y = df[['medv']]
x.head()
#전처리
pd.set_option('display.max_columns', None)
df.describe()
df.isnull().sum()
nf = [i for i in x.columns]

#min-max 스케일링
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# x[nf] = scaler.fit_transform(x[nf])

# iris = load_iris() #lord_iris import
# iris_data = iris.data #중요 데이터 iris_data 변수에 저장 후 데이터 크기 확인
# iris_label = iris.target #iris_label변수에 iris 데이터의 target저장
# print(iris.keys())
#
# df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
# if 'target_names' in iris.keys() :
#     df['target'] = iris.target_names[iris.target]
# else:
#     df['target'] = iris.target
# print(df)

#(3)train, test 데이터 분리
 #X:feature데이터만 / y:정답label데이터만
 #X데이터셋을 머신러닝 모델에 입력 -> 모델이 내뱉는 품종 예측 결과를 정답 y와 비교하여 학습

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3)
x_train.shape, x_test.shape, y_train.shape,y_test.shape
x_train
y_train
#라벨인코더

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
#X_train, X_test, y_train, y_test = train_test_split(iris_data,iris_label,test_size=0.3,random_state=7)
# feature(입력받는 특징 데이터)
# label(모델이 맞춰야하는 정답값)
# test dataset 크기 조절(0.2=전체 20%를 test데이터로 사용)
# train데이터와 test데이터 분리시 적용되는 랜덤성)



##################랜 덤 포 레 스 트
random_forest = RandomForestRegressor()
random_forest.fit(x_train, y_train)

rf_y_pred = random_forest.predict(x_test)
rf_y_pred
y_test
##시각화


plt.scatter(np.arange(len(rf_y_pred)),rf_y_pred, color = 'blue')

####중요변수 시각화########
ftr_importances_values = random_forest.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=x_train.columns)
ftr_top = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8, 6))
sns.barplot(x=ftr_top, y=ftr_top.index)
plt.show()
###########평가지표
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print(random_forest.score(x_test, y_test))
MSE = mean_squared_error(y_test, rf_y_pred)
np.sqrt(MSE) #rmse

r2_score(y_test,rf_y_pred)

########################################support vector machine
from sklearn import svm
from sklearn.svm import SVR

svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
svm_y_pred = svm_model.predict(x_test)

print(classification_report(y_test, svm_y_pred))

########################xgboost classifier###
xgboost_model = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4)
xgboost_model.fit(x_train,y_train)
xg_y_pred=xgboost_model.predict(x_test)
# accuracy_score(xg_y_pred, y_test)
print(classification_report(y_test, xg_y_pred))

#########################xgboost 회귀#####
xgboost_model2 = xgboost.XGBRegressor(learning_rate=0.1,max_depth=5,n_estimators=100)
xgboost_model2.fit(x_train,y_train)
xg_y_pred=xgboost_model2.predict(x_test)

MSE = mean_squared_error(y_test, xg_y_pred)
np.sqrt(MSE) #rmse
r2_score(y_test,xg_y_pred)
#sns.scatterplot(y_test,xg_y_pred)

#####################평 가 #####################################################
#############################################accuracy


accuracy_score(y_test, rf_y_pred)
accuracy_score(y_test, svm_y_pred)
accuracy_score(y_test, xg_y_pred)


###################################confusion_matrix, metrics
from sklearn.metrics import confusion_matrix  #오차행렬은 sklearn.metircs 패키지 내 confusion_matrix로 확인 가능

confusion_matrix(y_test, rf_y_pred)
confusion_matrix(y_test, svm_y_pred)
confusion_matrix(y_test, xg_y_pred)


################################## Precision, Recall, F1 score 지표 한눈에 확인
from sklearn.metrics import classification_report

print(classification_report(y_test, rf_y_pred))
print(classification_report(y_test, svm_y_pred))
print(classification_report(y_test, xg_y_pred))


#############회귀 시각화
import matplotlib.pyplot as plt
fig = plt.figure( figsize = (12, 4) )
chart = fig.add_subplot(1,1,1)
chart.plot(y_test[:100], marker='o', color='blue', label='real value')
chart.plot(y_pred[:100], marker='^', color='red', label='predict value')
chart.set_title('real value vs predict value')