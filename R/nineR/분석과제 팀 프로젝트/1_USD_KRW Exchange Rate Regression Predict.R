library(dplyr)
library(car)
library(caret)
library(lubridate)
library(lmtest)
library(ggplot2)
library(e1071)

daily <- read.csv("USD_KRW_day.csv", header=T, encoding = "UTF-8")
head(daily)
str(daily)
tail(daily)
colnames(daily)
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

######기본 가정 충족 확인#######
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

#######svr 회귀분석##############
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

#svr 예측
pred_svm<-predict(svm_model,newdata=fulldate[c(1:261),])
pred2_svm<-predict(svm_model,newdata=fulldate)
plot(y~x,xlim = c(0, 300),col='red',cex=1.5)
points(index(pred2_svm), pred2_svm, col = 'blue', pch = "*", cex = 1.2)
points(index(pred_svm), pred_svm, col = 'green', pch = "❤" ,cex = 1)
abline(reg=lm(y~x, data = df), col="black",lwd=1.5)
(pred_svm)
pred2<-predict(svm_model,newdata=head(fulldate,261))

R2(pred2,y)
rmse(pred2,y)


##########시계열 분석###############

#install.packages("forecast")
library(TTR)
library(forecast)

#1-2단계: 시계열 객체 생성
daily$date<-as.Date(daily$date)
#daily2<-daily%>%arrange(date)%>%mutate(pre_date=as.numeric(date-min(date)+1))
#daily2<-daily%>%arrange(date)%>%mutate(pre_date=as.numeric(rank(date)))
daily2 <- daily
head(daily2)
tsdaily <- ts(daily2$won, start=c(1), frequency = 1)
#pacf(na.omit(tsdaily), main = "자기 상환함수", col = "red")


#1-3단계: 추세선 시각화
plot(tsdaily, type = "l", col = "red")

#2단계: 정상성 시계열 변환
par(mfrow = c(1, 2))
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
