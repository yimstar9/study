#임성구
#################################################################################################
# [문항1]  * 아래 문제를 R code로 작성하여 제출하시오.
# 다음은 학생별 과목별 시험 점수이다. Data를 대상으로 데이터프레임을 생성하고, 그 데이터프레임을 사용하고apply()를 적용하여 행/열 방향으로 조건에 맞게 통계량을 구하시오.
# (난이도 : 3 / 배점 : 25점)

# # 1) 3명 의사의 과목점수를 이용하여 데이터프레임(DataFrame)을 생성하여 화면출력하시오.
# # 2) 수학과목에서 평균점수를 구하시오.
# # 3) 윤봉길의사의 과목 평균점수를 구하시오.
# # 4) 국어과목의 표준편차를 구하시오
# # 5) 각 과목의 최고점을 받은 사람은 누구인지 코딩하여 결과를 산출하시오.

sub <- c("국어(Kor)","영어(Eng)","수학(Mat)")
y <- c(95,83,75)
a <- c(85,95,60)
l <- c(70,80,95)
ans <- data.frame(과목 = sub, 윤봉길 = y, 안중근=a,이봉창=l)
mean_math <- apply(ans[ , c(2:4)], 1, mean)[3]
mean_Yoon <- mean(ans$윤봉길)
sd_kor <- apply(ans[ , c(2:4)], 1, sd)[1]
ans


# # 1) 3명 의사의 과목점수를 이용하여 데이터프레임(DataFrame)을 생성하여 화면출력하시오.
ans
# 과목 윤봉길 안중근 이봉창
# 1 국어(Kor)     95     85     70
# 2 영어(Eng)     83     95     80
# 3 수학(Mat)     75     60     95


# 2) 수학과목에서 평균점수를 구하시오.
mean_math
# 76.66667

# 3) 윤봉길의사의 과목 평균점수를 구하시오.
mean_Yoon
# 84.33333 


# 4) 국어과목의 표준편차를 구하시오
sd_kor
# 12.58306


# 5) 각 과목의 최고점을 받은 사람은 누구인지 코딩하여 결과를 산출하시오.
print("국어점수 최고점 받은 사람:");names(which.max(ans[1,c(2:4)]))
print("영어점수 최고점 받은 사람:");names(which.max(ans[2,c(2:4)]))
print("수학점수 최고점 받은 사람:");names(which.max(ans[3,c(2:4)]))


# [1] "국어점수 최고점 받은 사람:"
# [1] "윤봉길"
# [1] "영어점수 최고점 받은 사람:"
# [1] "안중근"
# [1] "수학점수 최고점 받은 사람:"
# [1] "이봉창


################################################################################################
# [문항2]  RSADBE 패키지에서 제공되는 Bug_Metrics_Software 데이터 셋을 대상으로 다음을 구하시오.
# (난이도: 3 / 배점: 20점)
# (1) 소프트웨어 발표 후 행 단위 평균을 구하시오 
# (2) 소프트웨어 발표 후 열 단위 합계를 구하시오 
# (3) 칼럼 단위로 요약통계량을 구하시오.
library(RSADBE)

data("Bug_Metrics_Software")
str(Bug_Metrics_Software)
Bug_Metrics_Software

# (1) 소프트웨어 발표 후 행 단위 평균을 구하시오
rowMeans(Bug_Metrics_Software[,,2])

# (2) 소프트웨어 발표 후 열 단위 합계를 구하시오 
colSums(Bug_Metrics_Software[,,2])

# (3) 칼럼 단위로 요약통계량을 구하시오.
apply(Bug_Metrics_Software[,,],2,summary)


################################################################################################
# [문항3]  다음과 같이 데이터프레임을 구성하였다.
# exam_data = data.frame(
#   name = c('Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'),
#   score = c(12.5, 9, 16.5, 12, 9, 20, 14.5, 13.5, 8, 19),
#   attempts = c(1, 3, 2, 3, 2, 3, 1, 1, 2, 1),
#   qualify = c('yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes')
# )
# 다음을 실행하는 R code를 작성하시오.
# (난이도: 4 / 배점: 30점)

# (1) 각 이름의 국적은 다음과 같다. 각 개인의 국적을 데이터프레임에 추가하고 데이터프레임을 화면 출력하시오.
# (2) 기존의 데이터프레임에 다음의 두 사람을 추가하고 업데이트 된 데이터프레임을 화면 출력하시오. 
# (3) 업데이트된 데이터프레임에서 Qualify 항목을 제외한 데이터프레임을 화면 출력하시오 
# (4) 업데이트된 데이터프레임에서 Dima와 Jona를 제외한 데이터프레임을 화면 출력하시오 
# (5) 업데이트된 데이터프레임에서 이름과 그들의 국적만 화면 출력하시오


 exam_data = data.frame(
   name = c('Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'),
   score = c(12.5, 9, 16.5, 12, 9, 20, 14.5, 13.5, 8, 19),
   attempts = c(1, 3, 2, 3, 2, 3, 1, 1, 2, 1),
   qualify = c('yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes')
 )
exam_data

# (1) 각 이름의 국적은 다음과 같다. 각 개인의 국적을 데이터프레임에 추가하고 데이터프레임을 화면 출력하시오.
exam_data$country <- c("RUS","CHN","USA","USA","USA","USA","USA","USA","USA","USA")
exam_data

# name score attempts qualify country
# 1  Anastasia  12.5        1     yes    RUS
# 2       Dima   9.0        3      no    CHN
# 3  Katherine  16.5        2     yes    USA
# 4      James  12.0        3      no    USA
# 5      Emily   9.0        2      no    USA
# 6    Michael  20.0        3     yes    USA
# 7    Matthew  14.5        1     yes    USA
# 8      Laura  13.5        1      no    USA
# 9      Kevin   8.0        2      no    USA
# 10     Jonas  19.0        1     yes    USA



# (2) 기존의 데이터프레임에 다음의 두 사람을 추가하고 업데이트 된 데이터프레임을 화면 출력하시오. 
a <- data.frame(name="Kim", score=15, attempts=1,qualify="yes", country="KOR")
b <- data.frame(name="Lee", score=10, attempts=3,qualify="no", country="KOR")
exam_data <- rbind(exam_data,a)
exam_data <- rbind(exam_data,b)
exam_data

# name score attempts qualify country
# 1  Anastasia  12.5        1     yes     RUS
# 2       Dima   9.0        3      no     CHN
# 3  Katherine  16.5        2     yes     USA
# 4      James  12.0        3      no     USA
# 5      Emily   9.0        2      no     USA
# 6    Michael  20.0        3     yes     USA
# 7    Matthew  14.5        1     yes     USA
# 8      Laura  13.5        1      no     USA
# 9      Kevin   8.0        2      no     USA
# 10     Jonas  19.0        1     yes     USA
# 11       Kim  15.0        1     yes     KOR
# 12       Lee  10.0        3      no     KOR



# (3) 업데이트된 데이터프레임에서 Qualify 항목을 제외한 데이터프레임을 화면 출력하시오 
library(dplyr)
exam_data%>%select(-qualify)

# name score attempts country
# 1  Anastasia  12.5        1     RUS
# 2       Dima   9.0        3     CHN
# 3  Katherine  16.5        2     USA
# 4      James  12.0        3     USA
# 5      Emily   9.0        2     USA
# 6    Michael  20.0        3     USA
# 7    Matthew  14.5        1     USA
# 8      Laura  13.5        1     USA
# 9      Kevin   8.0        2     USA
# 10     Jonas  19.0        1     USA
# 11       Kim  15.0        1     KOR
# 12       Lee  10.0        3     KOR



# (4) 업데이트된 데이터프레임에서 Dima와 Jona를 제외한 데이터프레임을 화면 출력하시오 
exam_data%>%subset(!(exam_data$name=="Dima" | exam_data$name=="Jonas"))

# name score attempts qualify country
# 1  Anastasia  12.5        1     yes     RUS
# 3  Katherine  16.5        2     yes     USA
# 4      James  12.0        3      no     USA
# 5      Emily   9.0        2      no     USA
# 6    Michael  20.0        3     yes     USA
# 7    Matthew  14.5        1     yes     USA
# 8      Laura  13.5        1      no     USA
# 9      Kevin   8.0        2      no     USA
# 11       Kim  15.0        1     yes     KOR
# 12       Lee  10.0        3      no     KOR

# (5) 업데이트된 데이터프레임에서 이름과 그들의 국적만 화면 출력하시오
df <-  select(exam_data,name,country)
df

# name country
# 1  Anastasia     RUS
# 2       Dima     CHN
# 3  Katherine     USA
# 4      James     USA
# 5      Emily     USA
# 6    Michael     USA
# 7    Matthew     USA
# 8      Laura     USA
# 9      Kevin     USA
# 10     Jonas     USA
# 11       Kim     KOR
# 12       Lee     KOR



################################################################################################
# [문항4]  dplyr패키지와 iris 데이터 넷을 대상으로 아래의 문제를 실행하는 R코드를 작성하여 제출하시오
# (난이도 : 3 / 배점 : 25점)
# (1) iris의 꽃받침의 폭(Sepal.Width)이 3.7 이상의 값만 필터링하여 화면출력하시오.# 
# (2) (1)의 결과에서 2, 4, 5번째 컬럼을 선택하시오# 
# (3) (2)의 결과에서 2번 컬럼의 값에서 4번 컬럼의 값을 뺀 diff파생변수를 만들고, 앞부분 10개만 출력하시오# 
# (4) (3)의 결과에서 꽃의 종(Species)별로 그룹화하여 Sepal.Width와 Petal.Width 변수의 평균을 계산하시오.# 
# (5) (3)의 결과에서 위에서 4번째 꽃의 종(Species)는 무엇인지 알 수 있도록 코딩하시오.
library(dplyr)
data("iris")
# (1) iris의 꽃받침의 폭(Sepal.Width)이 3.7 이상의 값만 필터링하여 화면출력하시오.# 
n1 <- iris %>% filter(Sepal.Width >= 3.7);n1

# (2) (1)의 결과에서 2, 4, 5번째 컬럼을 선택하시오# 
n2 <- n1%>%select(c(2,4,5));n2

# (3) (2)의 결과에서 2번 컬럼의 값에서 4번 컬럼의 값을 뺀 diff파생변수를 만들고, 앞부분 10개만 출력하시오# 
n3 <- n2 %>% mutate(diff = Sepal.Width - Petal.Width);head(n3,n=10)

# (4) (3)의 결과에서 꽃의 종(Species)별로 그룹화하여 Sepal.Width와 Petal.Width 변수의 평균을 계산하시오.# 
n4 <- n3 %>% group_by(Species)%>%summarise(Sepal_mean=mean(Sepal.Width),Petal_mean=mean(Petal.Width));n4

# (5) (3)의 결과에서 위에서 4번째 꽃의 종(Species)는 무엇인지 알 수 있도록 코딩하시오.
n3$Species[4]

