# xgboost
# 1단계: 패키지 설치
#install.packages("xgboost",type="binary")
library(xgboost)
library(caret)
# xgboost 패키지
# 2단계: y변수 생성
iris_label <- ifelse(iris$Species == 'setosa', 0,
                     ifelse(iris$Species == 'versicolor', 1, 2))
table(iris_label)
iris$label <- iris_label
# 3단계: data set 생성
set.seed(1000)
idx <- sample(nrow(iris), 0.7 * nrow(iris))
train <- iris[idx, ] 
test <- iris[-idx, ]
# 4단계: matrix 객체 변환
train_mat <- as.matrix(train[-c(5:6)])

dim(train_mat)
train_lab <- train$label
length(train_lab)
train_lab

# x변수는 matrix 객체로 변환. y변수는 label을 이용하여 설정
# 5단계: xgb.DMatrix 객체 변환
dtrain <- xgb.DMatrix(data = train_mat, label = train_lab)
# xgb.DMatrix()함수: 학습데이터 생성
# https://www.rdocumentation.org/packages/xgboost/versions/1.3.2.1/topics/xgb.DMatrix
# 6단계: model생성 – xgboost matrix 객체 이용
xgb_model <- xgboost(data = dtrain, max_depth = 2, eta = 1,
                     nthread = 2, nrounds = 2,
                     objective = "multi:softmax", 
                     num_class = 3,
                     verbose = 0)
xgb_model
# xgboost()함수: 트리모델 생성
# https://www.rdocumentation.org/packages/xgboost/versions/0.4-4/topics/xgboost
# 7단계: test set 생성
test_mat <- as.matrix(test[-c(5:6)])
dim(test_mat)
test_lab <- test$label
length(test_lab)
# 8단계: model prediction
pred_iris <- predict(xgb_model, test_mat)
pred_iris
# predict()함수
# https://www.rdocumentation.org/packages/xgboost/versions/1.3.2.1/topics/predict.xgb.Booster
# 9단계: confusion matrix
table(pred_iris, test_lab)
# 10단계: 모델 성능평가1 – Accuracy
(12 + 15 + 16) / length(test_lab)
test_lab<-as.factor(test_lab)
pred_iris<-as.factor(pred_iris)
str(test_lab)
str(pred_iris)
caret::confusionMatrix(test_lab, pred_iris)$overall[1]

caret::confusionMatrix(test_lab, pred_iris)$byClass[7]
# 분류정확도
# 11단계: model의 중요 변수(feature)와 영향력 보기
importance_matrix <- xgb.importance(colnames(train_mat), 
                                    model = xgb_model)
importance_matrix
# xgb.importance() 함수
# https://www.rdocumentation.org/packages/xgboost/versions/0.6-4/topics/xgb.importance
# 12단계: 중요 변수 시각화
xgb.plot.importance(importance_matrix)

