##실습 (단순 선형 회귀분석 수행)
product <- read.csv("part3/product.csv", header = TRUE)
str(product)
product
#2단계: 독립변수와 종속변수 생성
y=product$제품_만족도
x=product$제품_적절성
df<-data.frame(x,y)

#단순선형회귀분석은 lm()함수 이용
# 형식: lm(formula = Y ~ X, data)
# Where
# X: 독립변수
# Y: 종속변수
# data: 데이터프레임

#3단계: 단순 선형회귀 모델 생성
result.lm<-lm(formula=y~x,data=df)

#4단계: 회귀분석의 절편과 기울기
result.lm

#5단계: 모델의 적합값과 잔차 보기
names(result.lm)

#5-1단계: 적합값 보기

fitted.values(result.lm)[1:2]

#5-2단계: 관측값 보기
head(df,1)

# 5-3단계: 회귀방정식을 적용하여 모델의 적합값 계산
Y = 0.7789 + 0.7393 * 4
Y

# 5-4단계: 잔차(오차) 계산
3 - 3.735963

# 5-5단계: 모델의 잔차 보기
# residuals: 모델의 잔차
residuals(result.lm)[1:2]

# 5-6단계: 모델의 잔차와 회귀방정식에 의한 적합값으로부터 관측값 계산

-0.7359630 + 3.735963
