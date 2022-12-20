#test

# 데이터 출처 : https://archive.ics.uci.edu/ml/datasets/Breast%20Cancer%20Wisconsin%20(Diagnostic)에서 변형
# 데이터 설명 :  유방암 발생여부 예측 (종속변수 diagnosis : B(양성)  , M(악성) )
# 문제타입 : 분류유형
# 평가지표 : f1-score
# trainData url : https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/train.csv
# testData url : https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/test.csv
# subData url : https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/submission.csv
#install.packages("caret")
#install.packages("caret", dependencies = c("Depends", "Suggests"))

library(dplyr)
library(caret)
library(car)
library(ModelMetrics)
library(randomForest)


train <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/train.csv",header=T)
test <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/test.csv",header=T)
subdata <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/submission.csv",header=T)

summary(train)

#변수 스케일링
minmax<-preProcess(train,"range")
train$perimeter_se <- predict(minmax, train)$perimeter_se
train$area_se<- predict(minmax, train)$area_se
train$area_worst<- predict(minmax, train)$area_worst
train$diagnosis<-as.factor(train$diagnosis)
#결측치 제거
colSums(is.na(train))
#train,test 분리
idx <- sample(1:nrow(train), 0.7*nrow(train))
df_train <- train[idx,]
df_test <- train[-idx,]
str(train)
m <- randomForest(diagnosis~.-(id), data=df_train, ntree=100) #랜덤포레스트
p <- predict(m,newdata=df_test)

# F1 score
caret::confusionMatrix(df_test$diagnosis, p)$byClass[7]


# Accuracy
caret::confusionMatrix(df_test$diagnosis, p)$overall[1]

#ROC auc
auc(df_test$diagnosis, p) #랜덤포레스트

str(test)
m2 <- randomForest(diagnosis~.-(id), data=train, ntree=100)
p2 <- predict(m2, newdata=test,type="prob")
length(p2)
result<-as.data.frame(as.character(p2))

colnames(result)<-c("diagnosis")
# head(result)
write.csv(result, "diagnosis.csv", row.names=F)

abc<-read.csv("diagnosis.csv")

head(abc)


