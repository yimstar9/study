#임성구
#ch15 연습문제(회귀분석)
#1. product.csv 파일의 데이터를 이용하여 다음의 단계별로 다중 회귀분석을 수행하시오.
product<-read.csv('part3/product.csv', header = TRUE)

#1단계: 변수 모델링
x <-sample(1:nrow(product), 0.7 * nrow(product))
train <- product[x, ]
test <- product[-x, ]
y = product$제품_만족도
x1 = product$제품_적절성
x2 = product$제품_친밀도
df <- data.frame(x1, x2, y)

#2단계: 학습데이터 이용 회귀모델 생성
library(car)
#product데이터 셋으로 다중 회귀분석
result.lm <- lm(formula = y ~x1 + x2, data = df)
result.lm
#(다중 공선성(Multicollinearity)문제 확인)
vif(result.lm)
sqrt(vif(result.lm)) > 3
cor(product)
summary(result.lm)
#다중공선성 문제가 발생하는 변수 없으므로 그대로 학습데이터 이용하여 회귀모델 생성
model <- lm(formula = 제품_만족도 ~제품_적절성 + 제품_친밀도, data = train)
summary(model)
#회귀방정식 
model
Y=0.8127 + 0.6529*x1 + 0.0941*x2
head(train,1)
Y=0.8127 + 0.6529*4 + 0.0941*4
Y
#잔차(오차)계산
Y-4
residuals(model)[1]

#3단계: 검정데이터 이용 모델 예측치 생성
pred <- predict(model,test)
pred

#4단계: 모델 평가: cor()함수 이용
cor(pred, test$제품_만족도)
#상관관계 0.77로 분류 정확도가 높다고 볼 수 있다.

#2. ggplot2 패키지에서 제공하는 diamonds 데이터 셋을 
#대상으로 carat, table, depth 변수 중에서 다이아몬드의 
#가격(price)에 영향을 미치는 관계를 다중회귀 분석을 이용하여
#예측하시오.

#1단계: 변수 모델링
library(ggplot2)
data(diamonds)
diamonds
idx <-sample(1:nrow(diamonds), 0.7 * nrow(diamonds))
diatrain <- diamonds[idx, ]
diatest <- diamonds[-idx, ]
price = diamonds$price
carat = diamonds$carat
table = diamonds$table
depth = diamonds$depth
df <- data.frame(carat, table, depth, price)


#2단계: 학습데이터 이용 회귀모델 생성
#diamonds데이터 셋으로 다중 회귀분석
result2.lm <- lm(formula = price ~carat + table + depth, data = df)
result2.lm
#(다중 공선성(Multicollinearity)문제 확인)
vif(result2.lm)
sqrt(vif(result2.lm)) > 2
summary(result2.lm)
cor(diamonds[,c(1,5,6,7)])
######조건1: 다이아몬드 가격 결정에 가장 큰 영향을 미치는 변수는?
# Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13003.441    390.918   33.26   <2e-16 ***
#   carat        7858.771     14.151  555.36   <2e-16 ***
#   table        -104.473      3.141  -33.26   <2e-16 ***
#   depth        -151.236      4.820  -31.38   <2e-16 ***
#가격에 가장 큰 영향을 미치는 변수는 carat이다.
#가격과 +관계 변수는 carat(7858.771) 이고 
#       -관계 변수는 table(-104.473),depth(-151.236)이다.


#다중공선성 문제가 발생하는 변수 없으므로 그대로 학습데이터 이용하여 회귀모델 생성
model2 <- lm(formula = price ~ carat+table+depth, data = diatrain)
summary(model2)
model2
Y=12923.6 + 7837.9*carat - 101.4*table -152.6*depth
head(diatrain,1)
Y=12923.6 + 7837.9*1.13 - 101.4*57 -152.6*60.1
Y
#잔차(오차)계산
Y-12150
residuals(model2)[1]

#3단계: 검정데이터 이용 모델 예측치 생성
pred2 <- predict(model2,diatest)
pred2

#4단계: 모델 평가: cor()함수 이용
cor(pred2, diatest$price)
#상관관계 0.9235로 분류 정확도가 높다고 볼 수 있다.
