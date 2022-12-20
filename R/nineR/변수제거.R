# 
# 변수 제거
# 1. 주성분 분석
# 2. 0에 가까운 분산을 가지는 변수 제거
# 분산이 0에 가까운 변수는 제거해도 큰 영향이 없음.
# nearZeroVar()함수
# https://www.rdocumentation.org/packages/caret/versions/6.0-86/topics/nearZeroVar
# where
# 'saveMetrics=FALSE'속성: 예측변수의 컬럼위치에 해당하는 정수 벡터
# 'saveMetrics=TRUE'속성: 컬럼을 가지는 데이터프레임
# freqRatio: 가장 큰 공통값 대비 두번째 큰 공통값의 빈도의 비율
# percentUnique: 데이터 전체로 부터 고유 데이터의 비율
# zeroVar: 예측변수가 오직 한개의 특이값을 갖는지 여부에 대한 논리 벡터
# nzv: 예측변수가 0에 가까운 분산예측 변수인지 여부에 대한 논리 벡터
# 실습.
# =======================
  # install.packages("caret")
library(caret) 
# install.packages("mlbench")
library(mlbench) 
nearZeroVar(iris, saveMetrics=TRUE)

data(Soybean)
head(Soybean)
# 0에 가까운 분산을 가지는 변수의 존재 여부 확인
nearZeroVar(Soybean, saveMetrics=TRUE)
# ======================
  # nzv = 'TRUE' 인 leaf.mild, mycelium, sclerotia 변수를 제거 해도 큰 영향이 없다.

# 3. 상관관계가 높은 변수 제거
# 상관관계가 높은 컬럼을 제외
# findCorrelation()함수
# https://www.rdocumentation.org/packages/caret/versions/6.0-88/topics/findCorrelation
# 실습.
# ============
  library(caret) 
library(mlbench)
data(Vehicle)
head(Vehicle)
# 상관관계 높은 열 선정
findCorrelation(cor(subset(Vehicle, select=-c(Class))))
# 상관관계가 높은 열끼리 상관관계 확인
cor(subset(Vehicle, select=-c(Class))) [c(3,8,11,7,9,2), c(3,8,11,7,9,2)]
# 상관관계 높은 열 제거
Cor_Vehicle <- Vehicle[,-c(3,8,11,7,9,2)]
findCorrelation(cor(subset(Cor_Vehicle, select=-c(Class))))
head(Cor_Vehicle)
# ======================
# 
# 4. 카이 제곱 검정을 통한 중요 변수 선발
# 카이제곱검정을 실행하여 중요 변수 선발
# 실습.
# =================
install.packages("FSelector")
install.packages("mlbench")
install.packages("rjava")
install.packages("c://rJava_1.0-4.tar.gz",repos=NULL,type="source")
Sys.setenv(JAVA_HOEM='C:/Program Files/Java/jdk-19')
library(FSelector)
library(mlbench)
data(Vehicle) 
#카이 제곱 검정으로 변수들의 중요성 평가
(cs <- chi.squared(Class ~., data=Vehicle))
#변수 중에서 중요한 5개 선별
cutoff.k(cs,5)
# ===================
# 
# 머신러닝 기반 데이터 분석 pp95-106 차원축소 참고