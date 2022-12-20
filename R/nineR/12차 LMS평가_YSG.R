#임성구
##############################Q1#######################################
#1.MASS 패키지에 있는 Animals 데이터 셋에 대해 R의 
#  기본 함수를 이용하여 body컬럼을 대상으로 다음의 기술통계량을 
#  구하시오. 코드와 답을 기재하세요.(난이도: 3, 배점: 20점)
#  (1) Animals 데이터 셋 구조 보기
#  (2) 요약통계량
#  (3) 평균
#  (4) 표준편차
#  (5) Animals 데이터 셋의 빈도수 구하기

library(MASS)
data(Animals)

# (1)Animals 데이터 셋 구조 보기
str(Animals)
# 'data.frame':	28 obs. of  2 variables:
# $ body : num  1.35 465 36.33 27.66 1.04 ...
# $ brain: num  8.1 423 119.5 115 5.5 ...

# (2) 요약통계량
summary(Animals$body)
# Min.  1st Qu.   Median     Mean  3rd Qu. Max. 
# 0.02     3.10    53.83  4278.44   479.00 87000.00 

# (3) 평균
mean(Animals$body)
# 4278.439

# (4) 표준편차
sd(Animals$body)
# 16480.49

# (5) Animals데이터 셋의 빈도수 구하기
table(Animals$body)
table(Animals$brain)

# > table(Animals$body)
# 0.023  0.12 0.122  0.28  1.04  1.35   2.5   3.3 
# 1     1     1     1     1     1     1     1 
# 6.8    10 27.66    35 36.33 52.16  55.5    62 
# 1     1     1     1     1     1     1     1 
# 100 187.1   192   207   465   521   529  2547 
# 1     1     1     1     1     1     1     1 
# 6654  9400 11700 87000 
# 1     1     1     1 

# > table(Animals$brain)
# 0.4     1   1.9     3   5.5   8.1  12.1  25.6 
# 1     1     1     1     1     1     1     1 
# 50    56    70   115 119.5 154.5   157   175 
# 1     1     1     2     1     1     1     1 
# 179   180   406   419   423   440   655   680 
# 1     1     1     1     1     1     1     1 
# 1320  4603  5712 
# 1     1     1 



##############################Q2#######################################
#2.iris데이터를 이용하여 5겹 3회 반복하는 교차검정 
#  데이터를 샘플링 하시오.
library(cvTools)
data(iris)
cross <- cvFolds(nrow(iris), K=5, R=3) 
K=1:5
R=1:3
cross
table(cross$which) 
for(r in R){
  cat('R=',r,'회\n')
  for(k in K){
    datas_idx <- cross$subsets[cross$which==k,r]
    cat('K=',k,'검정데이터 \n')
    print(iris[datas_idx,])
    cat('K=',k,'훈련데이터 \n')
    print(iris[-datas_idx,])
  }
}


##############################Q3#######################################
# 3.descriptive.csv 파일내 데이터를 data변수에 담고 
#   다음의 기술통계량을 구하시오. (난이도: 4, 배점: 30점)
# 1) Cost 컬럼 내 데이터에서 1이상 11이하의 데이터만 추출하여 평균을 구하시오.
# 2) 1)번에서 추출한 데이터를 내림차순으로 정렬한 후 첫 10줄의 데이터를 보이시오
# 3) 1)번에서 추출한 데이터에서 3사분위수를 구하시오.
# 4) Cost2 컬럼을 생성하여 1이상 4미만의 데이터는 1로, 4이상 8미만의 데이터는 2로 8이상의 데이터는 3으로 범주화하여 저장하고 첫 10줄의 데이터를 보이시오.
# 5) 4)번에서 생성한 Cost2 컬럼내 데이터를 이용하여 파이 그래프로 시각화 하시오.
install.packages("dplyr")
library(dplyr)
library(plyr)
data <- read.csv('part3/descriptive.csv',header = T);data
str(data)

#(1)
data%>%filter(cost>=1&cost<=11)%>%summarise(mean(cost,na.rm=T))
#5.360558

#(2)
data%>%filter(cost>=1&cost<=11)%>%arrange(-cost)%>%head(10)
# resident gender age level cost type survey pass
# 1         5      2  45     3  7.9    1      3    1
# 2         5      2  45     3  7.9    1      2    1
# 3        NA      1  47     3  7.7    2     NA    2
# 4        NA      2  56     3  7.7    2      2    1
# 5         5      2  61     1  7.7    1      2    2
# 6         5      2  55     1  7.7    1      1    2
# 7         3      1  60     1  7.2    1     NA    2
# 8         3      1  61     3  7.2    1      3    2
# 9         3      1  64     1  7.1    1      3    1
# 10        1      1  45    NA  7.1    1     NA    2

#(3)
data%>%filter(cost>=1&cost<=11)%>%arrange(-cost)%>%summarise(quantile(cost, 3/4))
# 6.2

#(4)
cost_range = c(1, 4, 8, max(data$cost,na.rm=T)+1)
data%>%mutate(cost2=cut(cost, cost_range, right=F, labels = c("1", "2", "3")))%>%head(10)
# resident gender age level  cost type survey pass cost2
# 1         1      1  50     1   5.1    1      1    2     2
# 2         2      1  54     2   4.2    1      2    2     2
# 3        NA      1  62     2   4.7    1      1    1     2
# 4         4      2  50    NA   3.5    1      4    1     1
# 5         5      1  51     1   5.0    1      3    1     2
# 6         3      1  55     2   5.4    1      3   NA     2
# 7         2      2  56     1   4.1    1     NA    2     2
# 8         5      1  49     2 675.0   NA     NA    2     3
# 9        NA      1  49     1   4.4    1     NA    2     2
# 10        2      1  49     2   4.9    1      1    1     2

#(5)
data%>%mutate(cost2=cut(cost, cost_range, right=F, labels = c("1", "2", "3")))%>%select(cost2)%>%table()%>%pie()

##############################Q4#######################################
#4.twomethod2.csv 내 데이터 셋을 이용하여 교육방법(method)에
#  따른 시험성적(score)에 차이가 있는지 검정하시오.
#  (난이도: 4, 배점: 30점)
#  1) 결측치를 제거하여 데이터 전처리 하시오
#  2) 가설 설정
#  3) 적합한 검정 방법 선택
#  4) 가설 검정
#  5) 검정 결과

#(1)
method <- read.csv('twomethod2.csv',header = T)
method <- na.omit(method);method
a <- subset(method,method==1)
b <- subset(method,method==2)
a1 <- a$score
b1 <- b$score
length(a1)
length(b1)
mean(a1)
mean(b1)

#(2)
#H0:교육방법에 따른 두 집단간 시험성적(score)의 평균에 차이가 없다.
#H1:교육방법에 따른 두 집단간 시험성적(score)의 평균에 차이가 있다.

#(3)
#교육전 교육후 성적에 대한 차이 여부를 검정하는것이므로 
#두 집단 평균 차이 검정(독립 표본T검정)을 수행한다.

#(4)
#동질성 검정
var.test(a1,b1)
#p-value =0.9951로 동질성 분포와 차이가 없다. -> 모수 검정 방법 수행

#양측 검정
t.test(a1,b1)
#p-value = 6.57e-07로 두집단간 차이가 있다.

#단측 검정
t.test(a1,b1,alter="greater",conf.int=TRUE,conf.level = 0.95)
#p-value = 1
t.test(a1,b1,alter="less",conf.int=TRUE, conf.level = 0.95)
#p-value = 3.285e-07

#(5)
# 양측 검정을 실시한 결과 유의수준 0.05에서 귀무가설이 기각되었다.
# 따라서 "교육방법에 따른 두 집단간 시험성적의 평균에 차이가 있다." 라고 말할 수 있다.
# 단측 검정을 실시한 결과 (method=2)교육방법의 시험성적 평균이 (method=1)교육방법의 시험성적 평균보다 더 높은 것으로 나타났다.
# 즉 method=2 교육방법이 효과가 더 좋은 것으로 분석된다.