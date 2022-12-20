# 
# 베이지안(Baysian)
# 베이지안 확률 모델은 주관적인 추론을 바탕으로 만들어진 ‘사전확률’을 추가적인 관찰을
# 통한 ‘사후확률’로 업데이트하여 불확실성을 제거할 수 있다고 믿는 방법.
# 베이즈 정리는 posteriori확률을 찾는 과정이고 베이즈 추론을 MAP(Maximum a Posteriori)
# 문제라고 부르기도 한다.
# 실습.
# ================
# install.packages("e1071")
# install.packages("caret")
library(e1071)
data <- read.csv(file = "heart.csv", header = T)
head(data)
str(data)
library(caret)
set.seed(1234)


tr_data <- createDataPartition(y=data$AHD, p=0.7, list=FALSE)
#tr_data <- sample(1:nrow(data),0.7*nrow(data))


tr <- data[tr_data,]
te <- data[-tr_data,]
Bayes <- naiveBayes(AHD~. ,data=tr)
Bayes

predicted <- predict(Bayes, te, type="class")
table(predicted, te$AHD) 
AHD <- as.factor(te$AHD)
confusionMatrix(predicted, AHD)
