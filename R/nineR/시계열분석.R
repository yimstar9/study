################2. 시계열 자료분석
################ (비정상성 시계열을 정상성 시계열로 변경)
# 1단계: AirPassengers 데이터 셋 가져오기
data(AirPassengers)
# # 더 알아보기 (AirPassengers 데이터 셋)
# https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/AirPassengers
# 2단계: 차분 적용 – 평균 정상화
par(mfrow = c(1, 2))
ts.plot(AirPassengers)
diff <- diff(AirPassengers)
plot(diff)
# ts.plot()함수: 시계열 시각화
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/ts.plot
# diff()함수: 차분
# https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/diff
# 차분을 수행한 결과가 대체로 일정한 값을 얻으면 선형의 추세를 갖는다는 판단 가능
# 만약, 시계열에 계절성이 있으면 계절 차분을 수행하여 정상성 시계열로 변경
# 차분된 것을 다시 차분했을 때 일정한 값들을 보인다면 그 시계열 자료는 2차식의
# 추세를 갖는다고 판단
# 3단계: 로그 적용 – 분산 정상화
par(mfrow = c(1, 2))
plot(AirPassengers)
log <- diff(log(AirPassengers))
plot(log)
# log()함수
# https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/log

################3. 시계열 자료 시각화
################3.1 시계열 추세선 시각화
################(단일 시계열 자료 시각화)
data("WWWusage")
str(WWWusage)
# 더 알아보기(WWWusage 데이터 셋)
# https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/WWWusage

# 2단계: 시계열 자료 추세선 시각화
X11()
ts.plot(WWWusage, type = "l", col = "red")
# X11()함수: 새로운 창에서 시각화
# https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/x11

############(다중 시계열 자료 시각화)
# EuStockMarkets 데이터 셋 사용
# 1단계: 데이터 가져오기
data(EuStockMarkets)
head(EuStockMarkets)
# 2단계: 데이터프레임으로 변환
EuStock <- data.frame(EuStockMarkets)
head(EuStock)
# 3단계: 단일 시계열 자료 추세선 시각화(1,000개 데이터 대상)
X11()
plot(EuStock$DAX[1:1000], type = "l", col = "red")
# 4단계: 다중 시계열 자료 추세선 시각화(1,000개 데이터 대상)
plot.ts(cbind(EuStock$DAX[1:1000], EuStock$SMI[1:1000]),
        main = "주가지수 추세선")
# plot.ts()함수: 시계열 자료 plot
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/plot.ts

#############3.2 시계열 요소 분해 시각화
#############(시계열 요소 분해 시각화)
#1단계: 시계열 자료 준비
data <- c(45, 56, 45, 43, 69, 75, 58, 59, 66, 64, 62, 65, 
          55, 49, 67, 55, 71, 78, 71, 65, 69, 43, 70, 75, 
          56, 56, 65, 55, 82, 85, 75, 77, 77, 69, 79, 89)
length(data)
#2단계: 시계열 자료 생성 – 시계열 자료 형식으로 객체 생성
tsdata <- ts(data, start = c(2016, 1), frequency = 12)
tsdata
#3단계: 추세선 확인 – 각 요인(추세, 순환, 계절, 불규칙)을 시각적으로 확인
ts.plot(tsdata)
#4단계: 시계열 분해
plot(stl(tsdata, "periodic"))

#stl()함수: 하나의 시계열 자료를 대상으로 시계열 변동요인인 계절요소(seasonal), 
#추세(trend), 잔차(remainder)를 모두 제공
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/stl

#5단계: 시계열 분해와 변동요인 제거
m <- decompose(tsdata)
attributes(m)
plot(m)
par(mfrow = c(1, 1))
plot(tsdata - m$seasonal)
#decompose()함수: 시계열 분해
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/decompose
#6단계: 추세요인과 불규칙요인 제거
plot(tsdata - m$trend)
plot(tsdata - m$seasonal - m$trend)

#3.3 자기 상관 함수/부분 자기 상관 함수 시각화
# (시계열 요소 분해 시각화)
# 1단계: 시계열 자료 생성
input <- c(3180, 3000, 3200, 3100, 3300, 3200, 
           3400, 3550, 3200, 3400, 3300, 3700)
length(input)
tsdata <- ts(input, start = c(2015, 2), frequency = 12)
# 2단계: 자기 상관 함수 시각화
acf(na.omit(tsdata), main ="자기상관함수", col = "red")
# acf()함수
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/acf
# 점선은 유의미한 자기 상관관계에 대한 임계값을 의미
# 모든 시차(lag)가 파란 점선 안쪽에 있기 때문에 서로 이웃한 시점 간의 자기 상관성은
# 없는 것으로 해석

# 3단계: 부분 자기 상관 함수 시각화
pacf(na.omit(tsdata), main = "부분 자기 상관 함수", col = "red")
# pacf()함수: 
# https://www.rdocumentation.org/packages/forecast/versions/8.14/topics/Acf
# 자기 상관 함수에 의해서 주기 생성에는 어떤 종류의 시간 간격이 영향을 미치는가를
# 보여주고 있다.
# 모든 시차가 점선 안에 있기 때문에 주어진 시점 간의 자기 상관성은 없는 것으로 해석


# 3.4 추세 패턴 찾기 시각화
# 실습 (시계열 자료의 추세 패턴 찾기 시각화)
# 1단계: 시계열 자료 생성
input <- c(3180, 3000, 3200, 3100, 3300, 3200, 
           3400, 3550, 3200, 3400, 3300, 3700)
# 2단계: 추세선 시각화
plot(tsdata, type = "l", col = "red")
# 3단계: 자기 상관 함수 시각화
acf(na.omit(tsdata), main = "자기 상환함수", col = "red")
# 4단계: 차분 시각화
plot(diff(tsdata, differences = 1))

# 4.2 평활법
# (2) 지수평활법(Exponential Smoothing)
data <- c(45, 56, 45, 43, 69, 75, 58, 59, 66, 64, 62, 65, 
          55, 49, 67, 55, 71, 78, 71, 65, 69, 43, 70, 75, 
          56, 56, 65, 55, 82, 85, 75, 77, 77, 69, 79, 89)
length(data)
tsdata <- ts(data, start = c(2016, 1), frequency = 12)
tsdata
# 2단계: 평활 관련 패키지 설치
install.packages("TTR")
library(TTR)
# TTR패키지
# 3단계: 이동평균법으로 평활 및 시각화
par(mfrow = c(2, 2))
plot(tsdata, main = "원 시계열 자료")
plot(SMA(tsdata, n = 1), main = "1년 단위 이동평균법으로 평활")
plot(SMA(tsdata, n = 2), main = "2년 단위 이동평균법으로 평활")
plot(SMA(tsdata, n = 3), main = "3년 단위 이동평균법으로 평활")
par(mfrow = c(1, 1))
# SMA()함수
# https://www.rdocumentation.org/packages/TTR/versions/0.24.2/topics/SMA

# 5. ARIMA 모형 시계열 예측
# 5.1 ARIMA 모형 분석 절차
# 시계열 분석 절차
# 1단계: 시계열 자료 특성분석(정상성/비정상성)
# 2단계: 정상성 시계열 변환
# 3단계: 모형 식별과 추정
# 4단계: 모형 생성
# 5단계: 모형 진단(모형 타당성 검정)
# 6단계: 미래 예측(업무 적용)

# 5.2 정상성 시계열의 비계절형
# 실습 (계절성이 없는 정상성 시계열분석)
# 정상성 시계열은 대체로 평균을 중심으로 진폭이 일정하게 나타난다.
# 만약 비정상성 시계열이면 차분을 통해서 정상성 시계열로 바꾸는 작업이 필요
# 1단계: 시계열 자료 특성분석
# 1-1단계: 데이터 준비
input <- c(3180, 3000, 3200, 3100, 3300, 3200, 
           3400, 3550, 3200, 3400, 3300, 3700)
# 1-2단계: 시계열 객체 생성(12개월)
tsdata <- ts(input, start = c(2015, 2), frequency = 12)
tsdata
# ts()함수: 벡터 자료(input)를 대상으로 시계열 객체(tsdata)생성
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/ts
# 1-3단계: 추세선 시각화
plot(tsdata, type = "l", col = "red")
# 차분을 통해서 비정상성 시계열을 정상성 시계열로 변환
# 차분은 일반 차분과 계절 차분으로 구분
# 계절성을 갖는 경우에는 계절 차분을 적용
# 2단계: 정상성 시계열 변환
par(mfrow = c(1, 2))
ts.plot(tsdata)
diff <- diff(tsdata)
plot(diff)
# diff()함수
# https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/diff
# 1차 차분으로 정상화가 되지 않으면 2차 차분을 수행
# 3단계: 모델 식별과 추정
# install.packages("forecast")
library(forecast)
arima <- auto.arima(tsdata)
arima
# forecast 패키지 설치
# ARIMA(AR, Diff, MA) -> ARIMA(1, 1, 0)
# 한번 차분한 결과가 정상성 시계열의 ARMA(1,0)모형으로 식별
# AIC(Akaike’s Information Criterion): 모형의 적합도와 간명성을 동시에 나타내는 지수로
# 값이 적은 모형을 채택
# BIC(Bayesian Information Criterion): 이론적 예측력을 나타내는 지표
# ARIMA(p, d, q)모형의 정상성 시계열 변환 방법:
#   d=0이면, ARMA(p, q)모형이며 정상성을 만족
# p=0이면, IMA(d, q)모형이며 d번 차분하면 MA(q)모형을 따른다.
# q=0이면, IAR(p, d)모형이며, d번 차분하면 AR(p)모형을 따른다.
# 4단계: 모형 생성
model <- arima(tsdata, order = c(1, 1, 0))
model
# 모형의 적합성 검증을 위해서 잔차가 백색 잡음(white noise)인가를 살펴본다.
# 백색잡음: 모형의 잔차가 불규칙적이고, 독립적으로 분포된 경우를 의미
# 특정 시간 간의 데이터가 서로 관련성이 없다(독립적인 관계)
# 모형을 진단하는 기준:
#   1) 자기 상관 함수의 결과가 유의미한 시차가 없는 경우
# 2) 오차 간의 상관관계가 존재하는지를 검정하는 방법인 Box-Ljung검정에서 p값이
# 0.05이상인 경우
# 5단계: 모형 진단(모형의 타당성 검정)
# 5-1단계: 자기 상관 함수에 의한 모형 진단
tsdiag(model)
# tsdiag()함수
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/tsdiag
# 좋은 시계열 모형은 잔차의 ACF에서 자가 상관이 발견되지 않고, p-value값이
# 0.05이상으로 분포
#  현재 ARIMA모형은 매우 양호한 시계열 모형
# 5-2단계: Box-Ljung검정에 의한 잔차항 모형 진단
Box.test(model$residuals, lag = 1, type = "Ljung")
# Box-Ljung검정: 모형의 잔차를 이용하는 카이제곱 검정방법. 시계열 모형이 통계적으로
# 적절한지 검정. P-value가 0.05이상이면 모형이 통계적으로 적절
# 백색잡음과정: 시계열 모형이 적합하다면 잔차항은 서로 독립이고 동일한 분포를 따름
# 정상 시계열은 이러한 백색잡음과정으로부터 생성

# 6단계: 미래 예측(업무 적용)
fore <- forecast(model)
fore
par(mfrow = c(1, 2))
plot(fore)
model2 <- forecast(model, h = 6)
plot(model2)
# forecast()함수: 시계열의 예측치를 제공하는 함수
# https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/forecast

# 5.3 정상성 시계열의 계절형
# 실습 (계절성을 갖는 정상성 시계열분석)
# 1단계: 시계열 자료 특성분석
# 1-1단계: 데이터 준비
data <- c(55, 56, 45, 43, 69, 75, 58, 59, 66, 64, 62, 65, 
          55, 49, 67, 55, 71, 78, 61, 65, 69, 53, 70, 75, 
          56, 56, 65, 55, 68, 80, 65, 67, 77, 69, 79, 82,
          57, 55, 63, 60, 68, 70, 58, 65, 70, 55, 65, 70)
length(data)
# 1-2단계: 시계열 자료 생성
tsdata <- ts(data, start = c(2020, 1), frequency = 12)
tsdata
# 1-3단계: 시계열 요소 분해 시각화
ts_feature <- stl(tsdata, s.window = "periodic")
plot(ts_feature)
# Seasonal, trend, random요소 분해 시각화를 통해 분석
# 차분을 통해 비정상성 시계열을 정상성 시계열로 변환
# 2단계: 정상성 시계열 변환
par(mfrow = c(1, 2))
ts.plot(tsdata)
diff <- diff(tsdata)
plot(diff)

# 3단계: 모형 식별과 추정
library(forecast)
ts_model2 <- auto.arima(tsdata)
ts_model2
ARIMA(0, 1, 1)(1, 1, 0)[12]
# where
# (0, 1, 1): 차분차수 1, MA모형 차수1. 한번 차분한 결과가 정상성 시계열의 ARMA(0, 
# 1)모형으로 식별
# (1, 1, 0): 계절성을 갖는 자기 회귀(AR)모형 차수가 1. 계절성을 갖는 시계열
# [12]: 계절의 차수 12개월
# 4단계: 모형 생성
model <- arima(tsdata, c(0, 1, 1), seasonal = list(order = c(1, 1, 0)))
model
# 5단계: 모형 진단(모형 타당성 검정)
# 5-1단계: 자기 상관 함수에 의한 모형 진단
tsdiag(model)
# 5-2단계: Box-Ljung에 의한 잔차항 모형 진단
Box.test(model$residuals, lag = 1, type = "Ljung")
# 모형 진단을 통해서 적절하나 모형으로 판단되면 이 모형으로 가까운 미래를 예측하는데
# 이용
# 6단계: 미래 예측(업무 적용)
par(mfrow = c(1, 2))
fore <- forecast(model, h = 24)
plot(fore)
fore2 <- forecast(model, h = 6)
plot(fore2)
#   2년 예측, 6개월 예측
