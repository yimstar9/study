from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #(4)sklearn.tree패키지에 DecisionTreeClassifier import
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
#다만, 분류에서는 model.predict_proba( )로 뽑아서 roc-auc로 평가하고
# 회귀에서는 model.predict로 뽑아서 r2 score나 MSE등의 지표로 평가한다는 게 다를 것 같네요.

#(2)데이터 준비
iris = load_iris() #lord_iris import
iris_data = iris.data #중요 데이터 iris_data 변수에 저장 후 데이터 크기 확인
iris_label = iris.target #iris_label변수에 iris 데이터의 target저장
print(iris.keys())

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
if 'target_names' in iris.keys() :
    df['target'] = iris.target_names[iris.target]
else:
    df['target'] = iris.target
print(df)

#(3)train, test 데이터 분리
 #X:feature데이터만 / y:정답label데이터만
 #X데이터셋을 머신러닝 모델에 입력 -> 모델이 내뱉는 품종 예측 결과를 정답 y와 비교하여 학습
X_train, X_test, y_train, y_test = train_test_split(iris_data,iris_label,test_size=0.3,random_state=7)
# feature(입력받는 특징 데이터)
# label(모델이 맞춰야하는 정답값)
# test dataset 크기 조절(0.2=전체 20%를 test데이터로 사용)
# train데이터와 test데이터 분리시 적용되는 랜덤성)

####################의 사 결 정 나 무  (4)decision tree 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32) #decision_tree변수에 모델 저장
decision_tree.fit(X_train, y_train) #fit메서드로 모델 학습
y_pred = decision_tree.predict(X_test)




##################덤덤 reerdgh덤 랜 덤 포 레 트
random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))

########################################support vector machine
from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))



#####################평 가 #####################################################
#############################################accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy

###################################confusion_matrix, metrics
from sklearn.metrics import confusion_matrix  #오차행렬은 sklearn.metircs 패키지 내 confusion_matrix로 확인 가능

confusion_matrix(y_test, y_pred)

################################## Precision, Recall, F1 score 지표 한눈에 확인
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
