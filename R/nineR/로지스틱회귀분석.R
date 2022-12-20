#로지스틱 회귀분석
# 실습 (날씨 관련 요인 변수로 비(rain) 유무 예측)
#install.packages("ROCR")
#install.packages("lmtest")
library(car)
library(lmtest)
library(ROCR)
# 1단계: 데이터 가져오기
weather = read.csv("part4/weather.csv", stringsAsFactors = F)
dim(weather)
head(weather)
str(weather)
# 2단계: 변수 선택과 더미 변수 생성
weather_df <- weather[ , c(-1, -6, -8, -14)]
str(weather_df)
weather_df$RainTomorrow[weather_df$RainTomorrow == 'Yes'] <- 1
weather_df$RainTomorrow[weather_df$RainTomorrow == 'No'] <- 0
weather_df$RainTomorrow <- as.numeric(weather_df$RainTomorrow)
head(weather_df)

#결측치 확인 및 제거
apply(is.na(weather_df),2,sum)
weather_df<-na.omit(weather_df)
# X, Y변수 설정
# Y변수를 대상으로 더미 변수를 생성하여 로지스틱 회귀분석 환경 설정
# 3단계: 학습데이터와 검정데이터 생성(7:3비율)
idx <- sample(1:nrow(weather_df ), nrow(weather_df) * 0.7)
train <- weather_df[idx, ]
test <- weather_df[-idx, ]
# 4단계: 로지스틱 회귀모델 생성
weather_model <- glm(RainTomorrow ~ ., data = train, family = 'binomial', na.action=na.omit)
weather_model
summary(weather_model)

# glm()함수
# 형식: glm(y~x, data, family)
# Where
# family = ‘binomial’ 속성: y변수가 이항형
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/glm
# 로지스틱 회귀모델의 결과는 선형 회귀모델과 동일하게 x변수의 유의성 검정을 제공
# 하지만 F-검정 통계량과 모델의 설명력은 제공되지 않는다.
# 5단계: 로지스틱 회귀모델 예측치 생성
pred <- predict(weather_model, newdata = test, type = "response")
pred

result_pred <- ifelse(pred >= 0.5, 1, 0)
result_pred
table(result_pred)
# predict() 함수
# https://www.rdocumentation.org/packages/car/versions/3.0-10/topics/Predict
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/predict
# type=”response”속성: 에측 결과를 0-1사이의 확률값으로 예측치를 얻기 위해서 지정
# 모델 평가를 위해서 예측치가 확률값으로 제공되기 때문에 이를 이항형으로 변환하는 과정이
# 필요 -> ifelse()함수를 이용하여 예측치의 벡터변수(pred)를 입력으로 이항형의 벡터
# 변수(result_pred)를 생성
# 6단계: 모델평가 – 분류정확도 계산
table(result_pred, test$RainTomorrow)

# * 분류정확도는 데이터에 따라 상이
# 7단계: ROC(Receiver Operating Characteristic) Curve를 이용한 모델 평가
# ROCR 패키지 설치
pr <- prediction(pred, test$RainTomorrow)
prf <- performance(pr,   measure = "tpr", x.measure = "fpr")
plot(prf)
a<-performance(pr, measure = "auc")
a@y.values

