#임성구
#14차 lms 평가
#1
#(1)
response <- read.csv("response.csv", header=TRUE)
#(2)
library(dplyr)
#%>%mutate(새컬럼=case_when(조건1~"참값return",조건2~"참값reutrn",TRUE~값)
result<-response%>%mutate(job2=case_when(job==1~"1:학생",
                               job==2~"2:직장인",
                               job==3~"3:주부"),
                response2=case_when(response==1~"1:무응답",
                                    response==2~"2:낮음",
                                    response==3~"3:높음"))


#(3)
table(result$job2,result$response2)
#(4)
chisq.test(result$job2, result$response2)
#p-value=6.901e-12 귀무가설 기각
#(5)
#귀무가설(H0:직업 유형에 따른 응답정도에 차이가 없다)을 기각한다.
#그러므로 응답정도에 차이가 있다고 할 수 있다.

#2
#(1)
install.packages("mlbench")
library(mlbench)
attach(attitude)

#(2)
colSums(is.na(attitude))
result.lm<-lm(formula=rating~.,data=attitude)


#(3)
summary(result.lm)
#회귀모형 결정계수R-squared: 0.7326, 수정결정계수Adjusted R-squared:  0.6628
#회귀모형의 적합성p-value: 1.24e-05

#(4)
vif(result.lm) #다중공선성 문제 확인
sqrt(vif(result.lm)) > 2 #문제 없음
result.lm2<-step(result.lm,direction = "backward") #후진제거법 변수제거
summary(result.lm2)

#(5)
formula(result.lm2)
#rating ~ complaints + learning
result.lm2
#rating=9.8709+0.6435*complaints+0.2112*learning


#3
#(1)
data3<-read.csv("cleanData.csv",header=T)
head(data3)
#(2)
x<-data3$position
y<-data3$age3

#(3)
plot(x,y)
plot(x,y,abline(lm(y~x)))
#(4)
library(gmodels)
CrossTable(x,y, chisq = TRUE) 
#(5)
#나이와 직위 간의 관련성이 있다를 분석하기 위해서 217명을 표본으로 추출한후
#설문조사하여 교차분석과 카이제곱검정을 시행하였다. 분석결과 나이와 직위의
#관련성은 유의미한 수준에서 차이가 있는 것으로 나타났다.(카이제곱값 = 287.8957 ,
#p-value = 1.548058e-57 )
#따라서 귀무가설을 기각 할 수 있기때문에 나이와 직위간 관련성이 있다고 분석된다.

#4
#(1)
mtcars
#(2)
colSums(is.na(mtcars))
library(car)
library(lmtest)
library(ROCR)
df<-mtcars[,c("vs","mpg","am")]
idx <- sample(1:nrow(df), nrow(df) * 0.7)
df_train <- df[idx, ]
df_test <- df[-idx, ]
model <- glm(vs~.,data=df_train,family='binomial')
str(model)
model[3]
#(3)
summary(model)
#(4)
formula(model)
#vs=-19.8282+1.0755*mpg-6.5293*am
#(5)
#mpg=30,am=0 ; odds?
y=-19.8282+1.0755*30-6.5293*0
exp(y)
#251903.2

#5
#(1)
#https://nate9389.tistory.com/1635
#a:72 b:2 c:36
#d:30 e:9
#f:102 g:11