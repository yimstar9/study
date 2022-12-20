#임성구
"1. 제공된 response.csv 파일 내 데이터에서 작업 유형에 
따른 응답 정도에 차이가 있는가를 단계별로 검정하시오."

#(1) 파일 가져오기 (파일 내 데이터 저장)
df<-read.csv("response.csv")
head(df)


#(2)코딩 변경 – 리코딩 
#Job 컬럼: 1: 학생,  2: 직장인,  3: 주부
#Response 컬럼: 1: 무응답,  2: 낮음,  3: 높음
library(dplyr)
result<-df%>%mutate(Job2 = case_when(job==1~"1:학생",
                                     job==2~"2:직장인",
                                     job==3~"3:주부"),
                   Response2=case_when(response==1~"1:무응답",
                                       response==2~"2:낮음",
                                       response==3~"3:높음"))
head(result)
"  job response   Job2 Response2
1   1        1 1:학생  1:무응답
2   1        1 1:학생  1:무응답
3   1        1 1:학생  1:무응답
4   1        1 1:학생  1:무응답
5   1        1 1:학생  1:무응답
6   1        1 1:학생  1:무응답
"


#(3)교차 분할표 작성
table(result$Job2,result$Response2)
"
1:무응답 2:낮음 3:높음
1:학생         25     37      8
2:직장인       10     62     53
3:주부          5     41     59"


# (4) 동일성 검정
chisq.test(result$Job2, result$Response2)

"Pearson's Chi-squared test

data:  result$Job2 and result$Response2
X-squared = 58.208, df = 4,
p-value = 6.901e-12"


#(5) 검정 결과 해석
#p-value = 6.901e-12 < 0.05이므로 
#귀무가설(H0:직업 유형에 따른 응답정도에 차이가 없다)을 기각한다.
#그러므로 응답정도에 차이가 있다고 할 수 있다.




"======================================================================================"
"2.attitude 데이터를 이용하여 등급(rating)에 영향을 미치는 요인을 
회귀를 이용해 식별하고 후진제거법을 이용하여 적절한 변수 선택을 
하여 최종 회귀식을 구하시오."

#(1) 데이터 가져오기
data("attitude")
head(attitude)


#(2) 회귀분석 실시
colSums(is.na(attitude))#결측값 확인
result.lm<-lm(formula=rating~.,data=attitude) #회귀분석


#(3) 수행결과 산출 및 해석
summary(result.lm)

"Call:
lm(formula = rating ~ ., data = attitude)

Residuals:
     Min       1Q   Median       3Q      Max 
-10.9418  -4.3555   0.3158   5.5425  11.5990 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 10.78708   11.58926   0.931 0.361634    
complaints   0.61319    0.16098   3.809 0.000903 ***
privileges  -0.07305    0.13572  -0.538 0.595594    
learning     0.32033    0.16852   1.901 0.069925 .  
raises       0.08173    0.22148   0.369 0.715480    
critical     0.03838    0.14700   0.261 0.796334    
advance     -0.21706    0.17821  -1.218 0.235577    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 7.068 on 23 degrees of freedom
Multiple R-squared:  0.7326,	Adjusted R-squared:  0.6628 
F-statistic:  10.5 on 6 and 23 DF,  p-value: 1.24e-05"

#귀무가설(H0): 독립변수들이 등급에 영향을 미친하고 볼 수 없다.
#대립가설(H1): 독립변수들이 등급에 영향을 미친다고 볼 수 있다.
#유의수준0.1에서 유의한 독립변수는 complaints와 learning이다.
#회귀모형 결정계수R-squared: 0.7326, 수정결정계수Adjusted R-squared:  0.6628
#회귀모형의 적합성p-value: 1.24e-05 
#귀무가설을 기각 할 수 있으므로 독립변수들이 등급에 영향을 미친다고 볼 수 있다.



#(4) 후진제거법을 이용하여 독립변수 제거
result.lm2<-step(result.lm,direction = "backward") #후진제거법 변수제거
summary(result.lm2)
"Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)   9.8709     7.0612   1.398    0.174    
complaints    0.6435     0.1185   5.432 9.57e-06 ***
learning      0.2112     0.1344   1.571    0.128    
---"



#(5) 최종 회귀식
formula(result.lm2)
"rating ~ complaints + learning"
result.lm2
"rating=9.8709+0.6435*complaints+0.2112*learning"




"======================================================================================"
"3. 제공된 cleanData.csv 파일 내 데이터에서 나이(age3)와 
직위(position)간이 관련성을 단계별로 분석하시오."


#(1) 파일 가져오기(파일 내 데이터 저장)
cl<-read.csv("cleanData.csv")
head(cl)



"(2) 코딩 변경(변수 리코딩)
x <- data$position # 행 – 직위변수 이용
y <- data$age3 #열 – 나이 리코딩 변수 이용"
x <- cl$position
y <- cl$age3



"(3) 산점도를 이용한 변수간의 관련성 보기(plot(x,y)함수 이용)"
plot(x,y)
plot(x,y,abline(lm(y~x)))



#(4) 독립성 검정
library(gmodels)
CrossTable(x,y, chisq = TRUE) 



#(5) 결과 해석
#나이와 직위 간의 관련성이 있다를 분석하기 위해서 217명을 표본으로 추출한후
#설문조사하여 교차분석과 카이제곱검정을 시행하였다. 분석결과 나이와 직위의
#관련성은 유의미한 수준에서 차이가 있는 것으로 나타났다.(카이제곱값 = 287.8957 ,
#p-value = 1.548058e-57 )
#따라서 귀무가설을 기각 할 수 있기때문에 나이와 직위간 관련성이 있다고 분석된다.




"======================================================================================"
"4. mtcars 데이터에서 엔진(vs)을 종속변수로, 연비(mpg)와 변속기종류(am)를 
독립변수로 설정하여 로지스틱 회귀분석을 실시하시오."


# (1) 데이터 가져오기
data("mtcars")
head(mtcars)



# (2) 로지스틱 회귀분석 실행하고 회귀모델 확인
library(car)
library(lmtest)
library(ROCR)
colSums(is.na(mtcars)) #결측값 확인

df<-mtcars[,c("vs","mpg","am")] #필요한 컬럼만 추출

idx <- sample(1:nrow(df), nrow(df) * 0.7) #학습,검증 데이터 분리
df_train <- df[idx, ]
df_test <- df[-idx, ]

model <- glm(vs~.,data=df_train,family='binomial',na.action=na.omit) #로지스틱 회귀모델 생성 
pred <- predict(model, df_test, type = "response")

result_pred <- ifelse(pred >= 0.5, 1, 0)  #회귀모델 예측치 생성
result_pred

table(result_pred, df_test$vs) #분류 정확도 계산

model
"Coefficients:
(Intercept)          mpg           am  
    -19.833        1.077       -3.627  

Degrees of Freedom: 21 Total (i.e. Null);  19 Residual
Null Deviance:	    30.5 
Residual Deviance: 12.97 	AIC: 18.97"
 

# (3) 로지스틱 회귀모델 요약정보 확인
summary(model)

"Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.9927  -0.4551  -0.1289   0.4669   1.5517  
Coefficients:
            Estimate Std. Error z value
(Intercept) -19.8326    12.0942  -1.640
mpg           1.0770     0.6556   1.643
am           -3.6272     2.3278  -1.558
            Pr(>|z|)
(Intercept)    0.101
mpg            0.100
am             0.119

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 30.498  on 21  degrees of freedom
Residual deviance: 12.972  on 19  degrees of freedom
AIC: 18.972"



# (4) 로지스틱 회귀식
formula(model)
"vs ~ mpg + am"
"ln(odds)=-19.8326 + 1.0770*mpg - 3.6272*am"

# (5) mpg가 30이고 자동변속기(am=0)일 때 승산(odds)?
y=-19.8326 + 1.0770*30 - 3.6272*0
exp(y)
"262340.9"

"======================================================================================"
"5. 새롭게 제작된 자동차의 성능(주행거리(마일)/갤런)을 -30도, 0도, 30도의 기온
하에 성능을 측정하였다. 각 기온당 측정된 성능데이터의 수는 4개였다. 성능데이터
로부터 다음의 ANOVA 테이블을 구성하였다. 빈칸에 들어갈 숫자(정수)와 숫자를 계산한 
식을 제시하시오."

"(1)a:72 
 (2)b:2
 (3)c:36
 (4)d:30 
 (5)e:9
 (6)f:102 
 (7)g:11
"

