#임성구
# 1. descriptive.csv 데이터 셋을 대상으로 다음 조건에 맞게 빈도분석 및 기술통계량 분석을
# 수행하시오.
# 1) 명목척도 변수인 학교유형(type), 합격여부(pass)변수에 대해 빈도분석을 수행하고 결과를
# 막대 그래프와 파이 차트로 시각화
# 2) 비율척도 변수인 나이 변수에 대해 요약치(평균, 표준편차)와 비대칭도(왜도와 첨도) 
# 통계량을 구하고, 히스토그램을 작성하여 비대칭도 통계량 설명
# 3) 나이 변수에 대한 밀도분포 곡선과 정규분포 곡선으로 정규분포 검정

data <- read.csv("part3/descriptive.csv", header = TRUE)

# 1) 명목척도 변수인 학교유형(type), 합격여부(pass)변수에 대해 빈도분석을 수행하고 결과를
# 막대 그래프와 파이 차트로 시각화
x <- table(data$type)
par(mfrow = c(1, 2))
barplot(x)
pie(x)

y <- table(data$pass)
barplot(y)
pie(y)

# 2) 비율척도 변수인 나이 변수에 대해 요약치(평균, 표준편차)와 비대칭도(왜도와 첨도) 
# 통계량을 구하고, 히스토그램을 작성하여 비대칭도 통계량 설명
library(moments)
attach(data)
mean(age)
sd(age)
skewness(age)
kurtosis(age)
hist(age)

# 3) 나이 변수에 대한 밀도분포 곡선과 정규분포 곡선으로 정규분포 검정
hist(age, freq = F)
lines(density(age), col = 'blue')
range(age)
x <- seq(40, 70, 0.1)
curve(dnorm(x, mean(age), sd(age)), col = 'red', add = T)



# 2. MASS 패키지에 있는 Animals 데이터 셋을 이용하여 각 단계에 맞게 기술통계량을 구하시오.
# 1) MASS 패키지 설치와 메모리 로딩
# 2) R 의 기본함수를 이용하여 brain 컬럼을 대상으로 다음의 제시된 기술통계량 구하기
# (1) Animals 데이터 셋 차원 보기
# (2) 요약통계량
# (3) 평균
# (4) 중위수
# (5) 표준편차
# (6) 분산
# (7) 최대값
# (8) 최소


# 1) MASS 패키지 설치와 메모리 로딩
library(MASS)
data(Animals) 
head(Animals)
attach(Animals)

# 2) R 의 기본함수를 이용하여 brain 컬럼을 대상으로 다음의 제시된 기술통계량 구하기
dim(Animals)
summary(brain)
mean(brain)
median(brain)
sd(brain) 
var(brain) 
max(brain) 
min(brain)  