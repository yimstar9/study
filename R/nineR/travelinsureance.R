# 작업형2 기출 유형(심화)¶
# 본 문제는 변형한 심화 문제 입니다.
# 오리지널 3회 기출 유형을 보고 싶은 분은 아래 클래스-커리큘럼 탭에 무료공개
# (3회 작업형2)로 영상과 데이터셋을 올려놨어요!
# https://class101.net/products/467P0ZPH0lVX9FwFBDz7
# 여행 보험 패키지 상품을 구매할 확률 값을 구하시오
# 예측할 값(y): TravelInsurance (여행보험 패지지를 구매 했는지 여부 0:구매안함, 1:구매)
# 평가: roc-auc 평가지표
# data: t2-1-train.csv, t2-1-test.csv

######################## 제출 형식#########################
# id,TravelInsurance
# 0,0.3
# 1,0.48
# 2,0.3
# 3,0.83

# Baseline
# 3회 기출문제에서 데이터 셋을 편집해 조금 더 어렵게 만들었어요
# 결측치 추가
# Employment Type 컬럼에 카테고리 추가
# sample_submission 파일은 제공된 적 없음(3회 때 제출 형식에 대한 이슈가 있어 
# 제공하거나 제출 형식을 명확하게 설명할 가능성 있어 보임)
# 데이터셋 https://www.kaggle.com/competitions/big-data-analytics-certification/data?select=t2-1-test.csv
# 데이터셋 https://www.kaggle.com/competitions/big-data-analytics-certification/data?select=t2-1-train.csv
train <- read.csv("t2-1-train.csv",header=T)
test <- read.csv("t2-1-test.csv",header=T)
install.packages("c:\\randomForest_4.6-14.tar.gz",repos=NULL,type="source")
library(randomForest)

head(train)
summary(train)
str(train)

#결측치 확인, factor 변환
train$Employment.Type<-as.factor(train$Employment.Type)
train$GraduateOrNot<-as.factor(train$GraduateOrNot)
train$FrequentFlyer <- as.factor(train$FrequentFlyer)
train$EverTravelledAbroad <- as.factor(train$EverTravelledAbroad)
train$TravelInsurance <- as.factor(train$TravelInsurance)
colSums(is.na(train))
train$AnnualIncome <- ifelse(is.na(train$AnnualIncome),0,train$AnnualIncome)

idx<-sample(1:nrow(train),nrow(train)*0.7)
dftrain <- train[idx,]
dftest <- train[-idx,]
str(dftrain)
m <- randomForest(TravelInsurance~.-(id),data=dftrain,ntree=100)
p <- predict(m,newdata=dftest,type="response")

#F1-score
caret::confusionMatrix(dftest$TravelInsurance, p)$byClass[7]


# Accuracy
caret::confusionMatrix(dftest$TravelInsurance, p)$overall[1]

#ROC auc
auc(dftest$TravelInsurance, p) #랜덤포레스트


# test$Employment.Type<-as.factor(test$Employment.Type)
# test$GraduateOrNot<-as.factor(test$GraduateOrNot)
# test$FrequentFlyer <- as.factor(test$FrequentFlyer)
# test$EverTravelledAbroad <- as.factor(test$EverTravelledAbroad)
# test$AnnualIncome <- as.numeric(test$AnnualIncome)
# str(test)
# str(dftest)
# str(m2)
# sapply(train, levels)
# sapply(test, levels)
# levels(train$Employment.Type) <-  levels(test$Employment.Type)
# nrow(test)
# str(test)
# nrow(p2)
help(predict)
help("predict.randomForest")

m2 <- randomForest(TravelInsurance~.-c(id), data=train, ntree=100)
p2 <- predict(m2, newdata=test,type="prob")
str(p2[,1])
p2[,2]
a <- as.character(test$id)
b <- as.character(p2[,1])
c<-data.frame(ID=a,Pro=b)
head(c)
write.csv(c, "travelInsurance.csv", row.names = F)
abc<-read.csv("travelInsurance.csv")
abc
