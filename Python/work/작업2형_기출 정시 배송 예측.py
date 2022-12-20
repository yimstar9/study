# 데이터 설명 : e-commerce 배송의 정시 도착여부 (1: 정시배송 0 : 정시미배송)
# x_train: https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv
# y_train: https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv
# x_test: https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv
# x_label(평가용) : https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_test.csv
# 데이터 출처 :https://www.kaggle.com/datasets/prachi13/customer-analytics (참고, 데이터 수정)
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
pd.set_option('display.max_columns', None)


#데이터 로드
x_train_ = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv")
y_train_ = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv")
x_test_ = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv")
y_test_ = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_test.csv")


x_train_.describe()
x_train_.info()
y_train_.describe()
y_train_.info()

# 1. 원본 복사, 불필요열 삭제

x_train = x_train_.copy()
y_train = y_train_.copy()
x_test = x_test_.copy()
y_test = y_test_.copy()


#x_train = x_train.drop('ID', axis=1)
y_train = y_train.drop('ID', axis=1)
#x_test = x_test.drop('ID', axis=1)
y_test = y_test.drop('ID', axis=1)

# import matplotlib.pyplot as plt
# import seaborn as sns

# x_train.plot(kind='box', y='Customer_care_calls')
# x_train.plot(kind='box', y='Customer_rating')
# x_train.plot(kind='box', y='Cost_of_the_Product') # 정규화 필요
# x_train.plot(kind='box', y='Prior_purchases')
# x_train.plot(kind='box', y='Discount_offered')
# x_train.plot(kind='box', y='Weight_in_gms') # 정규화 필요


# 6. 최소 최대 정규화
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x_train['Cost_of_the_Product'] = mms.fit_transform(x_train['Cost_of_the_Product'].values.reshape(-1, 1))
x_test['Cost_of_the_Product'] = mms.transform(x_test['Cost_of_the_Product'].values.reshape(-1, 1))

x_train['Weight_in_gms'] = mms.fit_transform(x_train['Weight_in_gms'].values.reshape(-1, 1))
x_test['Weight_in_gms'] = mms.transform(x_test['Weight_in_gms'].values.reshape(-1, 1))

# 7. 모델링 전처리
from sklearn.model_selection import train_test_split
x_dummies = pd.get_dummies(pd.concat([x_train, x_test]))
x_train_dummies = x_dummies[:x_train.shape[0]]
x_test_dummies = x_dummies[x_train.shape[0]:]

y = y_train['Reached.on.Time_Y.N']
X_Train, X_test, Y_Train, Y_test = train_test_split(x_train_dummies, y, test_size = 0.3, stratify=y, random_state=42)


# 8. 모델링
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #(4)sklearn.tree패키지에 DecisionTreeClassifier import
from sklearn import svm
from sklearn.neural_network import MLPClassifier



decision_tree = DecisionTreeClassifier(random_state=42) #decision_tree변수에 모델 저장
decision_tree.fit(X_Train, Y_Train) #fit메서드로 모델 학습
pred_dc = decision_tree.predict_proba(X_test)

logist_model = LogisticRegression()
logist_model.fit(X_Train, Y_Train)
pred_glm = logist_model.predict_proba(X_test)

svm_model = svm.SVC()
svm_model.fit(X_Train, Y_Train)
pred_svm = svm_model.predict(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_Train, Y_Train)
pred_rf = rf.predict_proba(X_test)

clf = MLPClassifier(solver = "lbfgs", alpha = 1e-3, hidden_layer_sizes = (5,2), random_state = 42)
clf.fit(X_Train,Y_Train)
pred_clf=clf.predict_proba(X_test)

# 9. 모델 평가
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

pred_dc_bi = decision_tree.predict(X_test)
pred_glm_bi = logist_model.predict(X_test)
pred_rf_bi = rf.predict(X_test)
pred_svm_bi = svm_model.predict(X_test)
pred_clf_bi = clf.predict(X_test)

print("============accuracy=================")
print("decision_tree:",accuracy_score(Y_test,pred_dc_bi))
print("logistic:",accuracy_score(Y_test,pred_glm_bi))
print("randomforest:",accuracy_score(Y_test,pred_rf_bi))
print("svm:",accuracy_score(Y_test,pred_svm_bi))
print("clf:",accuracy_score(Y_test,pred_clf_bi))

print("============ROC auc==================")
print("decision_tree:",roc_auc_score(Y_test, pred_dc[:,1]))
print("logistic:",roc_auc_score(Y_test, pred_glm[:,1]))
print("randomforest:",roc_auc_score(Y_test, pred_rf[:,1]))
print("svm:",roc_auc_score(Y_test, pred_svm))
print("clf:",roc_auc_score(Y_test, pred_clf[:,1]))

#print(classification_report(Y_test, pred_rf_bi))
# 10. 제출
pred_test_prob = rf.predict_proba(x_test_dummies)
pred_test = rf.predict(x_test_dummies)
pd.DataFrame({'ID': y_test_['ID'], 'Reached.on.Time_Y.N': pred_test_prob[:,1]}).to_csv('0123.csv', index=False)
pd.DataFrame({'ID': y_test_['ID'], 'Reached.on.Time_Y.N': pred_test}).to_csv('1234.csv', index=False)
df = pd.read_csv('0123.csv')
df1 = pd.read_csv('1234.csv')
df.head()
df1.head()