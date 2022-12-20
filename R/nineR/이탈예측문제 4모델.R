#https://www.datamanim.com/dataset/03_dataq/typetwo.html
#분류
#서비스 이탈예측 데이터
#데이터 설명 : 고객의 신상정보 데이터를 통한 회사 서비스 이탈 예측 (종속변수 : Exited)


#install.packages("caret")
# install.packages("ModelMetrics")
#install.packages("randomForest")
# install.packages("rpart")
#install.packages("car")
# install.packages("lmtest")
# install.packages("ROCR")
# install.packages("dplyr")
library(dplyr)
library(caret)
library(car)
library(lmtest)
library(ROCR)
library(ModelMetrics)
library(randomForest)
library(rpart)
library(e1071)

dfx_train <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv",header=T, stringsAsFactors=T)
dfy_train <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv",header=T, stringsAsFactors=T)
dfx_test <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv",header=T, stringsAsFactors=T)
dfy_test <-read.csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_test.csv",header=T, stringsAsFactors=T)

# 결측치 확인
#apply(is.na(x_train),2,sum)
#apply(is.na(x_test),2,sum)
#apply(is.na(y_train),2,sum)

df<-merge(dfx_train,dfy_train,by='CustomerId')
df<-df[,c(3,6,7,8,9,10,11,12,13)]
dfx_test<-dfx_test[,c(3,6,7,8,9,10,11,12)]
dfy_test<-dfy_test[,2]
# dfy_test<-as.factor(dfy_test)


idx<-sample(1:nrow(df), 0.7*nrow(df))
#idx <- createDataPartition(df$Exited, p=0.7, list=F)
df_train<-df[idx,]
df_test<-df[-idx,]

df_train$Exited<-as.factor(df_train$Exited)
df_test$Exited<-as.factor(df_test$Exited)

m1 <- rpart(Exited~., data=df_train) #의사결정나무
m2 <- glm(Exited~., data=df_train, family="binomial") #로지스틱 회귀분석
m3 <- randomForest(Exited~., data=df_train, ntree=100) #랜덤포레스트
m4 <- svm(Exited~., data=df_train)#서포트벡터머신

pd1<-predict(m1,newdata=df_test)
pd2<-predict(m2,newdata=df_test, type = "response")
pd3<-predict(m3,newdata=df_test)
pd4<-predict(m4,newdata=df_test)

p11<-as.factor(round(pd1[,2]))
p22<-as.factor(round(pd2))
p33<-pd3
p44<-pd4

#F1score
# caret::confusionMatrix(df_test$Exited, p11)$byClass[7]
# caret::confusionMatrix(df_test$Exited, p22)$byClass[7]
# caret::confusionMatrix(df_test$Exited, p33)$byClass[7]
# caret::confusionMatrix(df_test$Exited, p44)$byClass[7]

#Accuracy
# caret::confusionMatrix(df_test$Exited, p11)$overall[1]
# caret::confusionMatrix(df_test$Exited, p22)$overall[1]
# caret::confusionMatrix(df_test$Exited, p33)$overall[1]
# caret::confusionMatrix(df_test$Exited, p44)$overall[1]
help("confusionMatrix")
print("모델별 auc")
auc(df_test$Exited, p11) #의사결정나무
auc(df_test$Exited, p22) #로지스틱 회귀분석
auc(df_test$Exited, p33) #랜덤포레스트
auc(df_test$Exited, p44) #서포트벡터머신


#최종모델링 예측결과 확인하기
#test 예측하기
df$Exited <- as.factor(df$Exited)
m <- randomForest(Exited~., data=df, ntree=100)
#p1 <- predict(m, newdata=dfx_test,type="prob")
p <- predict(m, newdata=dfx_test,type="response")
# head(p)
# str(p)

length(p)
result<-as.data.frame(as.character(p))
# head(result)
write.csv(result, "1234.csv", row.names=F)
abc<-read.csv("1234.csv")
abc

tmp_f<-caret::confusionMatrix(as.factor(dfy_test), as.factor(p))
auc<- auc(dfy_test, as.numeric(p))
print("====최종 모델링 예측결과====")
cat("auc=",auc)
tmp_f$byClass[7]
tmp_f$overall[1]
# 답안 제출 참고
# 아래 코드 변수명과 수험번호를 개인별로 변경하여 활용
# write.csv(변수명,'003000000.csv',row.names=F)