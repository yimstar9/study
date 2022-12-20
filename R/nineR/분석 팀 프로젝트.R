library(dplyr)
library(car)
library(caret)
library(lubridate)
library(lmtest)
library(ggplot2)
library(e1071)
library(readr)
getwd()
daily <- read_csv("USD_KRW_Daily.csv")
head(daily)
str(daily)
tail(daily)
colnames(daily)
#daily<-daily[,c(1,2,7)]
#colnames(daily)<-c('date','won','var')
daily<-daily[,-6]
colnames(daily)<-c('date','won','start','high','low','var')

daily$won <- gsub(",", "", daily$won)   
daily$start <- gsub(",", "", daily$start)   
daily$high <- gsub(",", "", daily$high)   
daily$low <- gsub(",", "", daily$low)   
daily$var <- gsub("%", "", daily$var)


daily$date<-as.Date(daily$date)
daily<-daily%>%arrange(date)%>%mutate(predate=date-max(date),rank=rank(date))

daily$won<-as.numeric(daily$won)
daily$start<-as.numeric(daily$start)
daily$high<-as.numeric(daily$high)
daily$low<-as.numeric(daily$low)
daily$var<-as.numeric(daily$var)

summary(daily)
head(daily)
daily%>%filter(var>1)

#회귀 분석 실시
y <- daily$won
x <- as.numeric(daily$rank)
x1 <- daily$start
x2 <- daily$high
x3 <- daily$low
df <- data.frame(x,x1, x2,x3, y)
model <- lm(formula = y~x+x1+x2+x3, data = df)
summary(model)


#다중 공선성(Multicollinearity)문제 확인
vif(model)
sqrt(vif(model))>3
cor(df)
#세 변수간 상관관계가 강하므로 상관관계가 높은 x1,x2,x3변수 제거 x(날짜)변수 한개로 단순회귀분석을 실시

model2 <- lm(formula = y~x, data = df)
summary(model2)

plot(model2$model$y~model2$model$x,cex=0.5)
abline(reg=lm(y~x, data = df), col="red",lwd=1.5)
#회귀식 : won=1.143e+03 + 1.006e+00 *days


#12월 예측 결과
#n=34 11월15일~12월31일까지 평일은 총34일(11월(12일),12월(22일)) 까지 예측
ydate<-data.frame(x=as.numeric(c(262:295)))
fulldate <- data.frame(x=as.numeric(c(1:295)))
pred<-predict(model2,newdata=ydate)
plot(pred)
pred2<-predict(model2,newdata=head(fulldate,261))
R2(pred2,y)
rmse(pred2,y)
#######################################기본 가정 충족 확인######################
#잔차 독립성 검정(더빈왓슨)
dwtest(model2)
#DW = 2.1918, p-value = 0.9197
#alternative hypothesis: true autocorrelation is greater than 0
#p-value가 0.05 이상이 되어 독립성이 있다고 볼 수 있다.

#등분산성 검정
plot(model2, which = 1)
#점점 분산이 커지는 분포이다

#잔차 정규성 검정
attributes(model)
res <- residuals(model)
shapiro.test(res)
shapiro.test(rstandard(model2))
par(mfrow = c(1, 2))
hist(res, freq = F)
qqnorm(res)
#W = 0.97307, p-value = 7.911e-05 <0.05이므로 정규성 만족
res <- residuals(model2)
shapiro.test(res)
par(mfrow = c(1, 2))
hist(res, freq = F)
qqnorm(res)


########################################svr 회귀분석##############
#refer https://yamalab.tistory.com/15
#https://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html
svm_model <- svm(formula = y~x, data = df)
summary(svm_model)
str(svm_model)

#####매개변수 최적화
#https://mopipe.tistory.com/39
# best_model <- tune.svm(y~x,df,gamma=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),cost=c(1:10),epsilon = c(0.1))
best_model <- tune.svm(y~x,df,gamma=c(1:10),cost=c(0.5,1),epsilon = c(0.1))
best_model
summary(best_model)

# - best parameters:
#   gamma cost epsilon
# 0.5    5     0.1

svm_model <- svm(formula = y~x, data = df, gamma=10,cost=1, epsilon=0.1)
summary(svm_model)
##########svr 예측

pred_svm<-predict(svm_model,newdata=fulldate[c(1:262),])
pred2_svm<-predict(svm_model,newdata=fulldate)
plot(y~x,xlim = c(0, 300),col='red',cex=1.5)
points(index(pred2_svm), pred2_svm, col = 'blue', pch = "*", cex = 1.2)
points(index(pred_svm), pred_svm, col = 'green', pch = "❤" ,cex = 1)
abline(reg=lm(y~x, data = df), col="black",lwd=1.5)
(pred_svm)
pred2<-predict(svm_model,newdata=head(fulldate,262))

R2(pred2,y)
rmse(pred2,y)


##########svr 회귀식 ---------모르겠다
# #refer https://kr.mathworks.com/help/stats/understanding-support-vector-machine-regression.html
# yi=w.k(xi,x)+b
# 
# #데이터에 대한 매개변수 W 및 b의 값은 각각 3.429 및 0.088입니다. 매개변수를 계산하는 R 코드는 다음과 같습니다.
# ##SVR 모델의 매개변수 계산
# # W의 값 찾기 
# W = t (svm_model$coefs ) %*% svm_model$SV 
# # b의 값 찾기 
# b = svm_model$rho
# b
# W


##############################################################################시계열 분석###############

#install.packages("forecast")
library(TTR)
library(forecast)

#1-2단계: 시계열 객체 생성
daily$date<-as.Date(daily$date)
#daily2<-daily%>%arrange(date)%>%mutate(pre_date=as.numeric(date-min(date)+1))
#daily2<-daily%>%arrange(date)%>%mutate(pre_date=as.numeric(rank(date)))
daily2 <- daily
head(daily2)
str(daily)
tsdaily <- ts(daily2$won, start=c(1), frequency = 1)
#pacf(na.omit(tsdaily), main = "자기 상환함수", col = "red")


#1-3단계: 추세선 시각화
plot(tsdaily, type = "l", col = "red")

#2단계: 정상성 시계열 변환
par(mfrow = c(1,2))
plot(diff(tsdaily, differences = 1))

#3단계: 모델 식별과 추정
library(forecast)
arima <- auto.arima(tsdaily)
arima

#4단계: 모형 생성
model <- arima(tsdaily, order = c(0, 1, 0))
str(model)

#5단계: 모형 진단(모형의 타당성 검정)
tsdiag(model)
Box.test(model$residuals, lag = 1, type = "Ljung")#pvalue가 0.69이므로 통계적으로유의하다
#6단계: 미래 예측(업무 적용)
model2 <- forecast(model, h = 34) 
model2

#n=34 11월15일~12월31일까지 평일은 총34일(11월(총12일),12월(총22일)) 예측
plot(model2)

####################################
#http://www.datamarket.kr/xe/index.php?mid=board_BoGi29&document_srl=22328&listStyle=viewer
#https://velog.io/@lifeisbeautiful/R-%EC%9C%A0%EB%B0%A9%EC%95%94-%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0-with-%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4

# 2. 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여 기법별 결과를
# 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)
library(caret)
library(ModelMetrics)
#install.packages("randomForest",type="binary")
library(randomForest)
library(rpart)
library(dplyr)
library(e1071)
library(car)
library(ggplot2)
library(nnet)
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
#   
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
#   
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
str(df)


# train데이터와 test데이터 2개의 집단으로 분리
# 랜덤으로 훈련 7: 테스트 3의 비율로 분리
set.seed(1)
samples <- sample(nrow(df), 0.7 * nrow(df))
train <- df[samples, ]
test <- df[-samples, ]

table(train$V2)
table(test$V2)


#모델생성

model_glm <- glm(V2~., data=train, family="binomial") #로지스틱 회귀분석
model_rf <- randomForest(V2~., data=train, ntree=300) #랜덤포레스트
model_svm <- svm(V2~., data=train)
#model_mnet <- nnet(V2 ~ ., data=train, size = 2)
model_rf

#예측
pd_glm <- predict(model_glm,newdata=test, type = "response")
pd_rf<-predict(model_rf,newdata=test, type = "response")

summary(pd_rf)
pd_svm<-predict(model_svm,newdata=test)
pd_svm
#pd_mnet<-predict(model_mnet,newdata=test, type="class")

#예측결과
pd_glm <- ifelse(pd_glm)
table(pd_glm,test$V2)
table(pd_rf,test$V2)
table(pd_svm,test$V2)
#table(pd_mnet,test$V2)
#pd_mnet <- as.factor(pd_mnet)


#예측점수
#F1-Score
#caret::confusionMatrix(test$V2, pd_glm)$byClass[7]
caret::confusionMatrix(test$V2, pd_rf)$byClass[7]
caret::confusionMatrix(test$V2, pd_svm)$byClass[7]
#caret::confusionMatrix(test$V2, pd_mnet)$byClass[7]

#Accuracy
caret::confusionMatrix(test$V2, pd_rf)$overall[1]
caret::confusionMatrix(test$V2, pd_svm)$overall[1]
#caret::confusionMatrix(test$V2, pd_mnet)$overall[1]

#Roc auc
auc(test$V2, pd_rf)
auc(test$V2, pd_svm)
#auc(test$V2, pd_mnet)
##################시각화

plot(model_rf,type="S",lwd=3)


help(plot)
#변수 시각화
dev.off()
library(gridExtra)
p1<-ggplot(df,aes(x=V2))+geom_bar()+labs(x="diagnosis")
p2<-ggplot(df,aes(V2,V10))+geom_jitter(col='gray')+geom_boxplot(alpha=.5)+labs(x="diagnosis",y="mean_concave_points")
p3<-ggplot(df,aes(V2,V30))+geom_jitter(col='gray')+geom_boxplot(alpha=.5)+labs(x="diagnosis",y="V30")
p4<-ggplot(df,aes(V10,V30))+geom_jitter(col='gray')+geom_smooth()+labs(x="mean_concave_points",y="V30")
gridExtra::grid.arrange(p1,p2,p3,p4,ncol=2)

#v10, v30 / v23,v25가 중요변수
randomForest::varImpPlot(model_rf) 
randomForest::importance(model_rf)
plot(train$V10, train$V30)
ggplot(test, aes(V10, V30))+ ggtitle("RandomForest Classification")+geom_point(aes(color=pd_rf),cex=3)
ggplot(test, aes(V10, V23))+ ggtitle("RandomForest Classification")+geom_point(aes(color=pd_rf),cex=3)

ggplot(test, aes(V10, V30))+ ggtitle("SVM Classification")+geom_point(aes(color=pd_svm),cex=3)
ggplot(test, aes(V10, V23))+ ggtitle("SVM Classification")+geom_point(aes(color=pd_svm),cex=3)
############################################################test##########################################################

#plot(model_svm,train)
plot(model_svm, test, V10 ~ V30, slice = list( V25 = 0.5, V23 = 0.85, V26=0.77))
plot(model_svm, test, V23 ~ V10, slice = list( V25 = 1, V30 = 1.14, V26=0.1))
help(plot.svm)
#############################################################################
# 3. mlbench패키지 내 BostonHousing 데이터셋을 대상으로 예측기법 2개를 적용하여
# 기법별 결과를 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는MEDV 또는CMEDV를사용
library(caret)
library(randomForest)
library(mlbench)
library(ModelMetrics)
library(e1071)

data("BostonHousing")
df <- BostonHousing
write.csv(df,'BostonHousing.csv',row.names = F)
ls(df)
head(df)
str(df)
summary(df)


#scale
pre_df<-df[,-c(4,14)]
head(pre_df)
summary(pre_df)
pre <- preProcess(pre_df,'range')
pre_df<-predict(pre,pre_df)
pre_df<-cbind(pre_df,df[,c(4,14)])
head(pre_df)
#train, test 셋 나누기
set.seed(1)
idx <- sample(1:nrow(pre_df),nrow(pre_df)*0.7)
train <- pre_df[idx,]
test <- pre_df[-idx,]

########모델 생성(선형회귀, svr, 랜덤포레스트)
m_lm=lm(medv~.,data=train)
m_svr <- svm(medv~., data=train)
m_rf=randomForest(medv~.,data=train,ntree=100,proximity=T)

p_lm=predict(m_lm,subset(test,select=c(-medv)))
p_svr=predict(m_svr,subset(test,select=c(-medv)))
p_rf=predict(m_rf,subset(test,select=c(-medv)),type="response")
summary(m_lm)
summary(m_svr)
m_rf
# p_lm
# p_svr
# p_rf
# 
# help(R2)

R2(p_lm,test$medv)
rmse(p_lm,test$medv)

R2(p_svr,test$medv)
rmse(p_svr,test$medv)


R2(p_rf,test$medv)
rmse(p_rf,test$medv)


x<-test$medv
y<-p_svr
y2<-p_rf

library(pracma)
plot(x, y,xlab = "",ylab ="", main="SVR Model",col="red",pch=1)
m= polyfit(x, y)[1]
b= polyfit(x, y)[2]
par(new=TRUE)
plot(x, m * x + b, 'l',axes=F,xlab='실제값',ylab='예측값',col='blue')


plot(x, y2,xlab = "",ylab ="", main="RandomForestRegressor Model",col="red")
m= polyfit(x, y2)[1]
b= polyfit(x, y2)[2]
par(new=TRUE)
plot(x, m * x + b, 'l',axes=F,xlab='실제값',ylab='예측값',col='blue')

#################################################################################
#4. 아래의 조건을 고려하여 군집분석을 실행하시오.
# (1) 데이터: ggplot2 패키지 내 diamonds 데이터
# (2) philentropy::distance() 함수 내 다양한 거리 계산 방법 중 Euclidian거리를 제외한
# 3개를 이용하여 거리 계산 및 사용된 거리에 대한 설명
# (3) 탐색적 목적의 계층적 군집분석 실행
# (4) 군집수 결정 및 결정 사유 설명
# (5) k-means clustering 실행
# (6) 시각화
# (7) 거리 계산 방법에 따른 결과 차이 비교
library(ggplot2)
library(cluster)
library(philentropy)
data(diamonds)

diamonds <- na.omit(diamonds)
set.seed(1000)
t <- sample(1:nrow(diamonds),100)
df <- diamonds[t,]
############################## #########
# dist <- dist(df, method = "manhattan")
# dist2 <- dist(df, method = "canberra")
# dist3 <- dist(df, method = "minkowski")

x = c(0, 0)
y = c(6,6)

x1 = c(0, 0, 1, 0, 1)
y1 = c(1, 1, 1, 0 ,0)


distance(rbind(x, y), method = "manhattan")
distance(rbind(x, y), method = "canberra")
distance(rbind(x, y), method = "chebyshev")
distance(rbind(x1, y1), method = "jaccard")
###################계층적 군집분석
help(daisy)
dist <- distance(df[,-c(2,3,4)], method="euclidean")
hc <- hclust(daisy(dist, metric = "euclidean"))
par(mfrow=c(4,1))
plot(hc,hang = -1)
rect.hclust(hc, k = 3, border ="red")
#######################################거리계산방법에 따른 차이
help(distance)
getDistMethods()
# dist <- distance(df[,-c(2,3,4)], method="manhattan")
# dist2 <- distance(df[,-c(2,3,4)], method = "canberra")
# dist3 <- distance(df[,-c(2,3,4)], method = "chebyshev")
# 
# hc <- hclust(daisy(dist, metric = "gower"),method="single")
# hc2 <- hclust(daisy(dist2, metric = "gower"),method="single")
# hc3 <- hclust(daisy(dist3, metric = "gower"),method="single")
dist <- distance(df[,-c(2,3,4)], method="manhattan")
dist2 <- distance(df[,-c(2,3,4)], method = "canberra")
dist5 <- distance(df[,-c(2,3,4)], method = "jaccard")

hc <- hclust(daisy(dist),method="single")
hc2 <- hclust(daisy(dist2),method="single")
hc5 <- hclust(daisy(dist5),method="single")

par(mfrow=c(3,1))

plot(hc,hang=-1)
rect.hclust(hc, k = 2, border ="red")

plot(hc2,hang=-1)
rect.hclust(hc2, k = 2, border ="red")

plot(hc5,hang=-1)
rect.hclust(hc5, k = 2, border ="red")


par(mfrow=c(1,2))
agn1 <- agnes(df, metric="manhattan", stand=TRUE)
plot(agn1)
rect.hclust(agn1, k = 5, border ="red")

#
agn2 <- agnes(df, metric="canberra", stand=TRUE)
plot(agn2)
rect.hclust(agn2, k = 5, border ="red")

#
agn3 <- agnes(df, metric="minkowski", stand=TRUE)
plot(agn3)
rect.hclust(agn3, k = 5, border ="red")


# 군집수 결정 및 결정 사유 설명
# 군집 수에 따른 집단 내 제곱합(within-groups sum of squares)의 그래프
# 
# 군집간의 개체간 거리의 제곱합 : 데이터가 얼마나 뭉쳐져있는지
# 뭉쳐져있는 값이 커서도 안되고 너무 작아서도 안됨, 각 객체마다 적절한 withiness를 가져야하며 tot.withiness의 산
# 점도를 그려 거기서 적절한 중간값을 찾는다.

#########엘보우 기법으로 최적 군집수 찾기
test_tot <- as.numeric()
for (i in 1:10){
  result <- kmeans(t,i)
  test_tot[i] <- result$tot.withinss
}
plot(c(1:10),test_tot,type='b')

#########실루엣계수로 최적 군집수 찾기
kmeans_result <- kmeans(t,3)
dist <- dist(t, method="euclidean")
sil = silhouette(kmeans_result$cluster,dist)
plot(sil)

avg_sil <- function(k, data) {
  result3 <- kmeans(data, centers = k)
  ss <- silhouette(result3$cluster, dist(data))
  avgSil <- mean(ss[, 3])
  return(avgSil)
}

kClusters <- 2:10
resultForEachK <- data.frame(k = kClusters, silAvg = rep(NA, length(kClusters))) 
for(i in 1:length(kClusters)){
  resultForEachK$silAvg[i] <- avg_sil(kClusters[i],t)
}
barplot(resultForEachK$k, resultForEachK$silAvg,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")



##########k-means
t <- sample(1:nrow(diamonds), 1000)
test <- diamonds[t, ]
mydia <- test[c("price", "carat", "depth", "table")]
head(mydia)

result2 <- kmeans(mydia,2)
names(result2)
result2$cluster
mydia$cluster <- result2$cluster
head(mydia)
cor(mydia[ , -5], method = "pearson")

plot(mydia[,-5], col=mydia$cluster)
plot(mydia$carat, mydia$price, col = mydia$cluster)
points(result2$centers[ , c("carat", "price")],col = c( 1, 2,3), pch = 9, cex = 10)

