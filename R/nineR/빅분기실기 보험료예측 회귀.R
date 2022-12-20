#데이터호출
train <- read.csv('train.csv')
test <- read.csv('test.csv')

#EDA(확인결과 스케일링 필요)
str(train)
summary(train)
str(test)
summary(test)

#결측치 확인(확인결과 결측치 없음)
colSums(is.na(train))
colSums(is.na(test))

#train, test set의 factor label확인(확인결과 label동일함)
levels(train$sex);levels(test$sex)
levels(train$smoker);levels(test$smoker)
levels(train$region);levels(test$region)

#(종속변수charges빼고)min-max 스케일링
library(caret)
p<-preProcess(train,'range')
train$age<-(predict(p,train))$age
train$bmi<-(predict(p,train))$bmi
train$children<-(predict(p,train))$children

test$age<-(predict(p,test))$age
test$bmi<-(predict(p,test))$bmi
test$children<-(predict(p,test))$children


#홀드아웃교차검증용 데이터 분리
idx=sample(1:nrow(train),nrow(train)*0.7)
x_train<-train[idx,]
x_valid<-train[-idx,]

# Random Forest 후보모델 생성
set.seed(1)
library(randomForest)
model1<-randomForest(charges~.,x_train,ntree=100)
pred1<-predict(model1,x_valid[,-7])
str(x_valid)

model2<-randomForest(charges~.,x_train,ntree=200,mtry=5)
pred2<-predict(model2,x_valid[,-7])

#후보모델 평가
library(ModelMetrics)
R2(pred1,x_valid$charges) # R2 : 0.8848536
rmse(pred1,x_valid$charges) # RMSE : 4522.243

R2(pred2,x_valid$charges) # R2 : 0.8920022
rmse(pred2,x_valid$charges) # RMSE : 4196.587


#최종모델 선택 model2(R2높고, RMSE 낮은)의 파라미터 선택
model<-randomForest(charges~.,test,ntree=200,mtry=5)
pred<-predict(model,test[,-7])


#최종모델평가, 결과 채점 (시험에서는 알 수 없음,시험관들이 채점하는부분)
R2(pred,test$charges) # 0.9584879
rmse(pred,test$charges) # 2411.521

#제출형식에 맞게 컬럼네임 변경, colnames()로 변경하는 경우도 있음
pred<-data.frame('charges'=pred) 

#csv파일 제출 및 확인
write.csv(pred,"result.csv", row.names = F)
confirm<-read.csv("result.csv")
head(confirm)
nrow(confirm);nrow(test)
