# mtcars 데이터는 1974년에 Motor Trend # US magazine의 데이터로,
# 32종류의 자동차를 10가지 항목을 조사한 데이터이며, 총 10가
# 지 항목은 다음과 같다.
#  mpg: 연비 (Miles/(US) gallon)
#  cyl: 실린더 개수 (Number of cylinders)
#  disp: 배기량 (Displacement (cu.in.))
#  hp: 마력 (Gross horsepower)
#  drat: 후방차축 비율 (Rear axle ratio)
#  wt: 무게 (Weight (1,000 lbs))
#  qsec: 1/4 마일에 도달하는데 소요되는 시간 (1/4 mile time)
#  vs: 엔진 (0 = V engine, 1 = S engine)
#  am: 변속기 (0 = 자동, 1 = 수동)
#  gear: 기어 개수 (Number of forward gears)
#  carb: 기화기 개수 (Number of carburetors)

#vs(엔진)를 종속변수로, mpg(연속형 독립변수)와 am(범주형 독립변수)으로 로지스틱 회귀분석 
#모델을 생성하고자 한다. 

data(mtcars) # mtcar datset 사용을 위한 코드

idx <- sample(1:nrow(mtcars),nrow(mtcars)*0.7)
dat <- subset(mtcars, select=c(mpg, am, vs)) #필요변수만 추출
x_train<-dat[idx,]
x_test<-dat[-idx,]
apply(is.na(dat),2,sum) #결측치 확인
dat

log_reg <- glm(vs ~ mpg+am, data=dat, family= "binomial") 
summary(log_reg)

# 5단계: 로지스틱 회귀모델 예측치 생성
pred <- predict(log_reg, newdata = x_test, type = "response")
pred
result_pred <- ifelse(pred >= 0.5, 1, 0)
result_pred
table(result_pred)
table(result_pred, x_test$vs)

pr <- prediction(pred, x_test$vs)
prf <- performance(pr,   measure = "tpr", x.measure = "fpr")
plot(prf)
a<-performance(pr, measure = "auc")
a@y.values
