#2.3(다중 회귀분석)
###실습 (다중 회귀분석)
product <- read.csv("part3/product.csv", header = TRUE)
#1단계: 변수 모델링
y = product$제품_만족도
x1 = product$제품_친밀도
x2 = product$제품_적절성
df <- data.frame(x1, x2, y)
#2단계: 다중 회귀분석
result.lm <- lm(formula = y ~x1 + x2, data = df)
result.lm
#실습 (다중 공선성(Multicollinearity)문제 확인)
#1단계: 패키지 설치
#install.packages("car")
library(car)
#car 패키지
#2단계: 분산팽창요인(VIF)
vif(result.lm)
#분산팽창요인(VIF)값이 10이상인 경우 다중 공선성 문제를 의심
#10이 절대값은 아님.

# 실습 (다중 회귀분석 결과보기)
summary(result.lm)
# 결과 제시 방법
# 가설, 분석결과, 가설검정, 회귀모형 결정계수, 수정결정계수, 회귀모형의 적합성, 
# 독립변수 설명

#2.4 다중 공선성 문제 해결과 모델 성능평가
library(car)
data(iris)
# car 패키지 설치
# 2단계: iris 데이터 셋으로 다중 회귀분석
model <- lm(formula = Sepal.Length ~ Sepal.Width + 
              Petal.Length + Petal.Width, data = iris)
vif(model)
sqrt(vif(model)) > 2
# 3단계: iris 변수 간의 상관계수 구하기
cor(iris[ , -5])
# 상관계수로 변수간의 강한 상관관계 구분

# (2) 회귀모델 생성
# 동일한 데이터 셋을 7:3 비율로 학습데이터와 검정데이터로 표본 추출한 후
# 학습데이터를 이용하여 회귀모델을 생성
# 실습 (데이터 셋 생성과 회귀모델 생성)

# 1단계: 학습데이터와 검정데이터 표본 추출
x <-sample(1:nrow(iris), 0.7 * nrow(iris))
train <- iris[x, ]
test <- iris[-x, ]
# sample()함수 이용하여 70% 데이터 추출하여 학습데이터, 나머지 데이터는 검정데이터로
# 설정
# 다중 공선성 문제가 발생하는 Petal.Width변수를 제거한 후 학습데이터를 이용하여
# 회귀모델 생성

# 2단계: 변수 제거 및 다중 회귀분석
model <- lm(formula = Sepal.Length ~ Sepal.Width + Petal.Length, data = train)
model
summary(model)

# (3) 회귀방정식 도출
# 절편, 기울기, 독립변수(x)의 관측치를 이용하여 회귀방정식을 도출출
# 실습 (회귀방정식 도출)
# 1단계: 회귀방정식을 위한 절편과 기울기 보기
model
# 2단계: 회귀방정식 도출
head(train, 1)
# *sample data에 따라서 회귀방정식은 상이할 수 있음.
# 다중 회귀방정식 적용(예시)
Y = 2.3826 + 0.5684 * 2.9 + 0.4576 * 4.6
Y
6.6 - Y

# (4) 예측치 생성
# 검정데이터를 이용하여 회귀모델의 예측치를 생성.
# 학습데이터에 의해 생성된 회귀모델을 검정데이터에 적용하여 모델의 예측치를 생성
# predict()함수
# 형식: predict(model, data)
# where
# model: 회귀모델(회귀분석 결과가 저장된 객체)
# data: 독립변수(x)가 존재하는 검정데이터 셋
# 실습 (검정데이터의 독립변수를 이용한 예측치 생성)
pred <- predict(model, test)
pred
# *데이터에 따라 예측치는 상이할 수 있음.

# (5) 회귀모델 평가
# 모델평가는 일반적으로 상관계수를 이용
# 모델의 예측치(pred)와 검정데이터의 종속변수(y)를 이용하여 상관계수(r)를 구하여
# 모델의 분류정확도를 평가한다.
# 상관관계가 높다면 분류정확도가 높다고 볼 수 있음.
# 실습 (상관계수를 이용한 회귀모델 평가)
cor(pred, test$Sepal.Length)

