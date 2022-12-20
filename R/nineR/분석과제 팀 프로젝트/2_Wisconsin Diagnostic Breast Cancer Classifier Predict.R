####################################
#http://www.datamarket.kr/xe/index.php?mid=board_BoGi29&document_srl=22328&listStyle=viewer
#https://velog.io/@lifeisbeautiful/R-%EC%9C%A0%EB%B0%A9%EC%95%94-%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0-with-%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4
# 2. 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여 기법별 결과를
# 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)
library(caret)
library(ModelMetrics)
library(randomForest)
library(rpart)
library(dplyr)
library(e1071)
library(car)
library(ggplot2)
library(ROCR)
#데이터 호출
df <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=F)
write.csv(df,'wdbc.csv',row.names = F)
df1<-head(df, 10)
write.csv(df1,"2번 dataset.csv")
summary(df)
str(df)
df<-df[,-1]
df$V2 <- as.factor(df$V2)

#데이터 설명
#https://gomguard.tistory.com/52 변수 설명
# Attribute Information:
 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# Ten real-valued features are computed for each cell nucleus:
  
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)

#결측치 확인 및 제거
colSums(is.na(df))
df <- na.omit(df)

#데이터 스케일링 min-max 스케일링으로 모든데이터 0~1로
scale<-preProcess(df,"range")
df <- predict(scale,df)


# factor형으로 변환
final[,1] <- as.factor(final[,1])
# 컬럼명을 label1로 명명
colnames(final)[1] <- "label1"
# final2 확인
str(final)


# train데이터와 test데이터 2개의 집단으로 분리
# 랜덤으로 훈련 7: 테스트 3의 비율로 분리
set.seed(1)
samples <- sample(nrow(df), 0.7 * nrow(df))
train <- df[samples, ]
test <- df[-samples, ]

table(train$V2)
table(test$V2)


#모델생성
model_rf <- randomForest(V2~., data=train, ntree=100) #랜덤포레스트
model_svm <- svm(V2~., data=train)


#예측
pd_rf<-predict(model_rf,newdata=test, type = "response")
pd_svm<-predict(model_svm,newdata=test)

#예측결과
table(pd_rf,test$V2)
table(pd_svm,test$V2)


#예측점수
#F1-Score
caret::confusionMatrix(test$V2, pd_rf)$byClass[7]
caret::confusionMatrix(test$V2, pd_svm)$byClass[7]


#Accuracy
caret::confusionMatrix(test$V2, pd_rf)$overall[1]
caret::confusionMatrix(test$V2, pd_svm)$overall[1]


#Roc auc
auc(test$V2, pd_rf)
auc(test$V2, pd_svm)


#v10, v30, v23가 중요변수
randomForest::varImpPlot(model_rf) 
randomForest::importance(model_rf)
plot(train$V10, train$V30)
ggplot(test, aes(V10, V30))+ ggtitle("RandomForest Classification")+geom_point(aes(color=pd_rf),cex=3)
ggplot(test, aes(V10, V23))+ ggtitle("RandomForest Classification")+geom_point(aes(color=pd_rf),cex=3)

ggplot(test, aes(V10, V30))+ ggtitle("SVM Classification")+geom_point(aes(color=pd_svm),cex=3)
ggplot(test, aes(V10, V23))+ ggtitle("SVM Classification")+geom_point(aes(color=pd_svm),cex=3)