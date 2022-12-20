#전진 선택법
install.packages("mlbench")
library(mlbench)
data("BostonHousing")
# 회귀
ss <- lm(medv ~ .,data=BostonHousing)
# 전진선택
ss1 <- step(ss, direction = "forward")
formula(ss1)

#2. 후진제거법 (Backward Elimination)
# 후진제거법
#=============
library(mlbench)
data("BostonHousing")
# 회귀
ss <- lm(medv ~ .,data=BostonHousing)
# 후진제거
ss2 <- step(ss, direction = "backward")
formula(ss2)

#실습3
#rating(등급)에 영향을 미치는 요인을 회귀를 이용해 식별
#종속변수 rating에 영향을 미치는 
#독립변수: complaints, privileges, learning, raises, critical, advance
data(attitude)
head(attitude)
# 회귀분석
model <- lm(rating~. , data=attitude)
# 수행결과
summary(model)
ss3<-step(model,direction='backward')
summary(ss3)

#3. 단계선택법(Stepwise Selection)
#실습1-3.
#================
# 단계적 선택방법
library(mlbench)
data("BostonHousing")
# 회귀
ss <- lm(medv ~ .,data=BostonHousing)
# 단계적선택
ss4 <- step(ss, direction = "both")
formula(ss4)
