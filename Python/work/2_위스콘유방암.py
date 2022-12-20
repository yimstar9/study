#from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #(4)sklearn.tree패키지에 DecisionTreeClassifier import
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost
#from xgboost import DMatrix
import pandas as pd
import seaborn as sns

#다만, 분류에서는 model.predict_proba( )로 뽑아서 roc-auc로 평가하고
# 회귀에서는 model.predict로 뽑아서 r2 score나 MSE등의 지표로 평가한다는 게 다를 것 같네요.

#(2)데이터 준비
df = pd.read_csv("wdbc.csv")
df.info()
df.head
#데이터 분리
x = df.drop(columns=['V1','V2'])
y = df[['V2']]
x.head()
#전처리
df.describe()
df.isnull().sum()
nf = [i for i in x.columns]

#min-max 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x[nf] = scaler.fit_transform(x[nf])

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

#라벨인코더

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
#X_train, X_test, y_train, y_test = train_test_split(iris_data,iris_label,test_size=0.3,random_state=7)
# feature(입력받는 특징 데이터)
# label(모델이 맞춰야하는 정답값)
# test dataset 크기 조절(0.2=전체 20%를 test데이터로 사용)
# train데이터와 test데이터 분리시 적용되는 랜덤성)

####################의 사 결 정 나 무  (4)decision tree 모델 학습 및 예측
decision_tree = DecisionTreeClassifier() #decision_tree변수에 모델 저장
decision_tree.fit(x_train, y_train) #fit메서드로 모델 학습
y_pred = decision_tree.predict(x_test)

print(classification_report(y_test, y_pred))


##################랜 덤 포 레 스 트
random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
rf_y_pred = random_forest.predict(x_test)

print(classification_report(y_test, rf_y_pred))

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
# xgboost_model2 = xgboost.XGBRegressor(learning_rate=0.1,max_depth=5,n_estimators=100)
# xgboost_model2.fit(x_train,y_train)
# xg_y_pred=xgboost_model2.predict(x_test)
# sns.scatterplot(y_test,xg_y_pred)

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

#############중요변수################

coef = pd.Series(random_forest.feature_importances_, index = x_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
#visualize feature
import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
imp_coef.plot(kind = 'barh', color = 'lightseagreen')
plt.title("Feature Importance")
plt.xlabel('Score')
plt.ylabel('Features')
plt.show()

########################rf 분류 시각화
plt.scatter(x_test['V10'],x_test['V30'], c=rf_y_pred, cmap='winter')
plt.title("RandomForest classification")
plt.xlabel('V10')
plt.ylabel('V30')
plt.colorbar(shrink=0.7)
plt.scatter(x_test['V10'],x_test['V25'], c=rf_y_pred, cmap='winter')
plt.title("RandomForest classification")
plt.xlabel('V10')
plt.ylabel('V25')
plt.colorbar(shrink=0.7)
########################svm 분류 시각화
plt.scatter(x_test['V10'],x_test['V30'], c=xg_y_pred, cmap='winter')
plt.title("XGB classification")
plt.xlabel('V10')
plt.ylabel('V30')
plt.colorbar(shrink=0.7)
plt.scatter(x_test['V10'],x_test['V25'], c=xg_y_pred, cmap='winter')
plt.title("XGB classification")
plt.xlabel('V10')
plt.ylabel('V25')
plt.colorbar(shrink=0.7)
#############회귀 시각화
# import matplotlib.pyplot as plt
# fig = plt.figure( figsize = (12, 4) )
# chart = fig.add_subplot(1,1,1)
# chart.plot(y_test[:100], marker='o', color='blue', label='real value')
# chart.plot(y_pred[:100], marker='^', color='red', label='predict value')
# chart.set_title('real value vs predict value')