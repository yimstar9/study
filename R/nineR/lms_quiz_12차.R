#임성구
#1
library(MASS)
data(Animals)
Animals
#(1)
str(Animals)
#(2)
summary(Animals$body)
#(3)
mean(Animals$body,na.rm=T)
#(4)
sd(Animals$body,na.rm=T)
#(5)
table(Animals$body)

#2
install.packages("cvTools")
library(cvTools)
data(iris)
cross <- cvFolds(nrow(iris), K=5, R=3) 
K=1:5
R=1:3
table(cross$which) 
for(r in R){
  cat('R=',r,'회\n')
  for(k in K){
    datas_idx <- cross$subsets[cross$which==k,r]
    cat('K=',k,'검정데이터 ]n')
    print(iris[datas_idx,])
    
    cat('K=',k,'훈련데이터 \n')
    print(iris[-datas_idx,])
  }
}

#3
getwd()
data <- read.csv('수업DATA/dataset2/descriptive.csv');data
table(data$cost)
#(1)
data1 <-data[(data$cost>=1&data$cost<=11),];data1
ans <- mean(data1$cost,na.rm=T)

#(2)
head(data1[order(-data1$cost),],10)

#(3)
summary(data1$cost)
quantile(data1$cost, 3/4,na.rm=T)

#(4)
rng = c(1, 4, 8, 12)
cost2 <-cut(data1$cost, rng, right=F, labels = c("1", "2", "3")) 
data1$cost2 <- cost2
head(data1,10)

#(5)
data1$cost
x1 <- table(data1$cost2)
pie(x1)


#4
#(1)
method <- read.csv('twomethod2.csv');method
method <- na.omit(method);method
#(2)
#H0:교육방법에 따른 두 집단간 시험성적(score)의 평균에 차이가 없다.
#H1:교육방법에 따른 두 집단간 시험성적(score)의 평균에 차이가 있다.
#(3)
#교육전 교육후 성적에 대한 차이를 검정하는것이므로 두 집단 평균 차이 검정(독립 표본T검정)
#(4)
a <- subset(method,method==1)
b <- subset(method,method==2)
a1 <- a$score
b1 <- b$score
length(a1)
length(b1)
var.test(a1,b1)#동질성 검정, p-value =0.9822로 동질성 분포와 차이가 없다. -> 모수 검정 방법 수행
t.test(a1,b1)#p-value = 6.57e-07로 두집단간 차이가 있다.
t.test(a1,b1,alter="greater",conf.int=TRUE,conf.level = 0.95)#p-value = 1
t.test(a1,b1,alter="less",conf.int=TRUE, conf.level = 0.95)#p-value = 3.285e-07
#(5)
# 유의수준 0.05에서 귀무가설이 기각되었다.
# 따라서 "교육방법에 따른 두 집단간 시험성적의 평균에 차이가 있다." 라고 말할 수 있다.
# 단측 검정을 실시한 결과 (method=2) 교육방법 평균이 (method=1) 교육방법 평균보다 시험성적이 더 좋은것으로 나타났다.
# 즉 method=2 교육방법이 효과가 더 높은 것으로 분석된다.