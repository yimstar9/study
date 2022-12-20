#임성구
# 1. 직업 유형에 따른 응답 정도에 차이가 있는가를 단계별로 검정하시오 (동질성 검정)
# 1) 파일 가져오기
#    Response.csv
# 2) 코딩변경 - 리코딩
#    Job 컬럼: 1. 학생, 2. 직장인, 3. 주부
#    response 컬럼: 1. 무응답, 2. 낮음. 3. 높음
# 3) 교차 분할표 작성
# 4) 동질성 검정
# 5) 검정결과 해석


#(1)
library(dplyr)
df<-read.csv('part3/Response.csv')
#(2)
df%>%mutate(job2=case_when(
  job == 1 ~"학생",
  job == 2 ~"직장인",
  job == 3 ~"주부"
))%>%mutate(response2=case_when(
  response == 1 ~"무응답",
  response == 2 ~"낮음",
  response == 3 ~"높음"
))

#(3)
t<-df%>%mutate(job2=case_when(
  job == 1 ~"학생",
  job == 2 ~"직장인",
  job == 3 ~"주부"
))%>%mutate(response2=case_when(
  response == 1 ~"무응답",
  response == 2 ~"낮음",
  response == 3 ~"높음"
))%>%select(job2,response2)%>%table()
t  
#(4) 동질성 검정  
chisq.test(t)
#p-value = 6.901e-12
#귀무가설(H0:세 집단의 응답률은 차이가 없다)를 기각한다.

#(5)
#세 집단 간의 응답율이 서로 다르다고 할 수 있다.





# 2. 나이(age)와 직위(position)간의 관련성을 단계별로 분석하시오 (독립성 검정)
# 1) 파일 가져오기
# cleanData.csv
# 2) 코딩 변경(변수 리코딩)
# X <- data$position #행: 직위 변수 이용
# Y <- data$age3 #열: 나이 리코딩 변수 이용
# 3) 산점도를 이용한 변수간의 관련성 보기
# (힌트. Plot(x,y)함수 이용)
# 4) 독립성 검정
# 5) 검정 결과 해석
#(1)
library(ggplot2)
library(gmodels)
df<-read.csv('part3/cleanData.csv');
str(df)
nrow(df)
#(2)
X <- df$position
Y <- df$age3
X
Y
#(3) 나이가 많음 (음의 상관관계)

#(4)
CrossTable(X,Y, chisq = TRUE) 

#(5)
#'나이와 직위는 관련성이 있다.'를 분석하기 위해서 217개를 표본으로 
#추출한 후 설문 조사하여 교차분석과 카이제곱검정을 실시하였다. 
#분석결과 (Chi^2 =  287.8957, p= 1.548058e-57, p<0.05)이므로 나이와 직위간의 
#관련성은 차이가 있는 것으로 나타났다.



# 3. 교육수준(education)과 흡연율(smoking)간의 관련성을 분석하기 위한 연구가설을 수립하고, 
# 단계별로 가설을 검정하시오. (독립성 검정)
#   귀무가설(H0): 
#   연구가설(H1):
#    1) 파일 가져오기
#     smoke.csv
#    2) 코딩변경
#     education 컬럼(독립변수) : 1. 대졸, 2. 고졸, 3. 중졸
#     smoke 컬럼(종속변수): 1. 과다흡연, 2. 보통흡연, 3. 비흡연
#    3) 교차분할표 작성
#    4) 검정 결과 해석

#(1)
df<-read.csv('part3/smoke.csv');
# 귀무가설(H0): 교육수준과 흡연율 간의 관련성은 없다. 
# 연구가설(H1): 교육수준과 흡연율 간의 관련성은 있다.

#(2)
t<-df%>%mutate(education2=case_when(
  education == 1 ~"대졸",
  education == 2 ~"고졸",
  education == 3 ~"중졸"
))%>%mutate(smoke2=case_when(
  smoking == 1 ~"과다흡연",
  smoking == 2 ~"보통흡연",
  smoking == 3 ~"비흡연"
))
t

#(3)
df%>%mutate(education2=case_when(
  education == 1 ~"대졸",
  education == 2 ~"고졸",
  education == 3 ~"중졸"
))%>%mutate(smoke2=case_when(
  smoking == 1 ~"과다흡연",
  smoking == 2 ~"보통흡연",
  smoking == 3 ~"비흡연"
))%>%select(education2,smoke2)%>%table()

#(4)
CrossTable(t$education2, t$smoke2, chisq = TRUE)

#p-value=0.003110976 <0.05 이므로 유의미한 수준에서 귀무가설 기각.
#‘교육수준과 흡연율 간의 관련성은 있다.’ 라고 볼 수 있다.