# 1. binary 분류문제에서 아래의 결과 데이터를 산출했다. 아래의 성능평가
# 지표를 산출하기 위하여 R 또는 python package 내 함수를 이용하여 coding하시오.
from sklearn import metrics
Y1_pred = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
Y1_true = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]

# (1) 정확도(accuracy)
metrics.accuracy_score(Y1_true,Y1_pred)
# 0.8

# (2) 정밀도(precision)
metrics.precision_score(Y1_true,Y1_pred)
# 1.0

# (3) 재현율(recall)
metrics.recall_score(Y1_true,Y1_pred)
# 0.6666666666666666

# (4) f1 score
metrics.f1_score(Y1_true,Y1_pred)
# 0.8

# (5) fbeta score (beta = 2)
metrics.fbeta_score(Y1_true,Y1_pred,beta=2)
# 0.7142857142857142

#################################################################
# 2. 다중분류문제에서 아래의 결과 데이터를 산출했다. 아래의 성능평가 지표를
# 산출하기 위하여 R 또는 python package 내 함수를 이용하여 coding하시오.
Y2_pred = ["cat", "dog", "cat", "cat", "dog", "bird", "bird"]
Y2_true = ["dog", "dog", "cat", "cat", "dog", "cat", "bird"]

from sklearn.metrics import classification_report
print(classification_report(Y2_true,Y2_pred))
#               precision    recall  f1-score   support
#         bird       0.50      1.00      0.67         1
#          cat       0.67      0.67      0.67         3
#          dog       1.00      0.67      0.80         3
#     accuracy                           0.71         7
#    macro avg       0.72      0.78      0.71         7
# weighted avg       0.79      0.71      0.72         7

# (1) 정확도(accuracy)
metrics.accuracy_score(Y2_true,Y2_pred)
# 0.7142857142857143

# (2) 정밀도(precision)
# average:{'micro','macro','samples','weighted','binary'}
# 'micro' : 총 TP,FN,FP을 이용해서 스코어를 계산
# 'macro' : 레이블의 가중되지 않은 평균
# 'weighted' : 각 레이블에 대한 실제 인스턴스의 수 별로 가중된 가중평균
# 'samples' : 오직 accuracy_score와 다른 다중분류에서만 의미가 있다.

metrics.precision_score(Y2_true,Y2_pred,average='macro')
# 0.7222222222222222

# (3) 재현율(recall)
metrics.recall_score(Y2_true,Y2_pred,average='macro')
# 0.7777777777777777

# (4) f1 score
metrics.f1_score(Y2_true,Y2_pred,average='macro')
# 0.7111111111111111

# (5) fbeta score (beta = 2)
metrics.fbeta_score(Y2_true,Y2_pred,beta=2,average='macro')
# 0.7380952380952381

###########################################################
# 3. Regression 문제에서 아래의 결과 데이터를 산출했다. 아래의 성능평가
# 지표를 산출하기 위하여 R 또는 python package 내 함수를 이용하여 coding하시오.
Y3_pred = [1, 8, 5, -2, -1, 2.5, 3, -0.5, 2, 7]
Y3_true = [0.8, 7.7, 5.2, -1.5, -0.8, 2.5, 3.2, 0.0, 2, 8]

# (1) 결정계수 R2
metrics.r2_score(Y3_true,Y3_pred)
# 0.9826576420339117

# (2) Mean absolute error(MAE)
metrics.mean_absolute_error(Y3_true,Y3_pred)
# 0.31

# (3) Mean squared error(MSE)
metrics.mean_squared_error(Y3_true,Y3_pred)
# 0.175

# (4) Mean absolute percentage error(MAPE)
metrics.mean_absolute_percentage_error(Y3_true,Y3_pred)
# 225179981368524.84