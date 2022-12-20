#임성구

# 1. 교육 방법에 따라 시험성적에 차이가 있는지 검정하시오
# (힌트. 두 집단 평균 차이 검정)
# 1) 데이터셋: twomethod.csv
# 2) 변수: method(교육방법), score(시험성적)

# 3) 모델: 교육방법(명목) -> 시험성적(비율)
# 4) 전처리, 결측치 제거

data <- read.csv("Part3/twomethod.csv", header=T);data

#4)결측치 제거
d2 <- subset(data, !is.na(score), c(method, score));d2

#3)두 집단 분리(모델)
m1 <- subset(d2,d2$method == 1);m1
m2 <- subset(d2,d2$method == 2);m2

m1score <- m1$score
m2score <- m2$score

#등분산성 검정
var.test(m1score,m2score)
#p-value=0.8494 > 0.05 이므로 
#귀무가설(H0:분산 차이가 없다)을 채택
#ratio of variances 는 1.06479임을 확인 할 수 있다.

#두 집단 간의 동질성 검정에서 분포의 형태가 동질하다고 분석 
#t.test()함수 이용 두 집단 간 평균 차이 검정

t.test(m1score, m2score) 
# p-value = 1.303e-06
#p-value가 0.05보다 작으니 통계적으로 유의하다

t.test(m1score, m2score, alter="greater", conf.int=TRUE, conf.level=0.95) 
#p-value = 1

t.test(m2score, m1score, alter="greater", conf.int=TRUE, conf.level=0.95) 
#p-value=6.513e-07
# m2 교육 방법이 m1 교육방법 보다 시험성적이 더 좋다.

################################################################
# 2. 대학에 진학한 남학생과 여학생을 대상으로 진학한 
# 대학에 대해서 만족도에 차이가 있는가를 검정하시오.
#(힌트. 두 집단 비율 차이 검정)
# 1) 데이터셋: two_sample.csv
# 2) 변수: gender(1,2), survey(0, 1)
getwd()
data <- read.csv("Part3/two_sample.csv", header=T);data

# 두 집단 subset 작성 
gender<- data$gender 
survey<- data$survey 

# 집단별 빈도수
table(gender) 
table(survey) 
table(gender, survey, useNA="ifany") 

# 두 집단 비율차이검증 
help("prop.test")
prop.test(c(138,107),c(174,126), alternative="two.sided", conf.level=0.95)
# p-value = 0.2765 > 0.05 이므로 
# 귀무가설(H0:만족도 차이가 없다) 채택 

###############################################################
# 3. 우리나라 전체 중학교 2 학년 여학생 평균 키가 148.5cm 로 알려진 상태에서 A 중학교 2 학년
# 전체 500 명을 대상으로 10%인 50 명을 표본으로 선정하여 표본평균 신장을 계산하고 모집단의
# 평균과 차이가 있는 지를 단계별로 분석을 수행하여 검정하시오.
# 1) 데이터셋: student_height,csv
# 2) height <- stheight$height
# 3) 기술통계량 평균 계산
# 4) 정규성 검정
# 5) 가설 검정
data <- read.csv("Part3/student_height.csv", header=T);data
height <- data$height;height

# 3)기술 통계량/결측치 확인
summary(height) 
meanheight <- na.omit(height);mean(meanheight)
 
# 4)정규성 검정
shapiro.test(meanheight) 
# p-value = 0.0001853 <0.05 이므로  
# 귀무가설(H0:정규분포를 따른다)기각
# 정규분포가 아니므로 비모수 검정을 실시
par(mfrow = c(1, 2))
hist(meanheight)
qqnorm(meanheight)
qqline(meanheight)

# 4)비모수검정
#가설검정 - 양측검정
wilcox.test(meanheight, mu=148.5, alter="two.side", conf.level=0.95) 
# p-value = 0.067 > 0.05 이므로 
#귀무가설(H0:표본평균과 모집단의 평균148.5와 차이가 없다) 채택. 


################################################################
#4. 중소기업에서 생산한 HDTV 판매율을 높이기 위해서 프로모션을 
#진행한 결과 기존 구매비율보다 15% 향상되었는지를 단계별로 
#분석을 수행하여 검정하시오.
library(prettyR) 
data <- read.csv("Part3/hdtv.csv", header=T);data
summary(data)
length(data$buy)
data

#귀무가설(H0):구매비율15%와 차이가 없다.
#대립가설(H1):구매비율15%와 차이가 있다.

# 3)빈도수 비율계산
freq(data$buy)
table(data$buy)
table(data$buy, useNA="ifany")

# 4)가설검정
binom.test(c(10,40), p=0.15, alternative="two.sided", conf.level=0.95)
#p-value = 0.321 > 0.05 이므로
# 귀무가설 (구매비율15%와 차이가 없다)채택 

# 방향성 단측가설 검정
binom.test(c(10,40), p=0.15, alternative="greater", conf.level=0.95)
#p-value=0.2089 > 0.05
binom.test(c(10,40), p=0.15, alternative="less", conf.level=0.95) 
#p-value = 0.8801 > 0.05
# 방향성 단측가설은 모두 p-value가 0.05보다 크므로 기각된다.