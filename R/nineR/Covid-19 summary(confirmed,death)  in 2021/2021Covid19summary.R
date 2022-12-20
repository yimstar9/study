library(dplyr)
library(stringr)
library(corrgram)
library(ggplot2)


###빼기 함수(NA값 처리후 뺄셈)###
sub_ <- function(x,y){
  return(ifelse(is.na(x),0,x)-ifelse(is.na(y),0,y))
}
###평균 함수(NA,이상 갯수 만큼 제외한 평균)###
avg_ <- function(x,y,z){
  return(round(x/(365-(y+z)),2))
}
getwd()
setwd('c:/nineR')
###2020년 데이터 로드###
rawData <- read.csv("covid19daily/12-31-2020.csv", header = T)
yesterdayDF <- rawData %>% group_by(Country_Region)%>% summarise(Confirmed=sum(Confirmed), Deaths=sum(Deaths))
tempDF <- yesterdayDF

###TEST데이터####
testData <- read.csv("covid19daily/12-31-2021.csv", header = T)
testData <- testData %>% group_by(Country_Region)%>% summarise(Confirmed=sum(Confirmed), Deaths=sum(Deaths))
testData <- merge(yesterdayDF,testData, by ='Country_Region', all = TRUE)
test_sub_DF <- testData%>% group_by(Country_Region) %>% summarise(Confirmed=sub_(Confirmed.y,Confirmed.x), Deaths=sub_(Deaths.y,Deaths.x))
test_sub_DF$T_Mean_C <- round(test_sub_DF$Confirmed/365,2)
test_sub_DF$T_Mean_D <- round(test_sub_DF$Deaths/365,2)
test_sub_DF

###2021년 파일 리스트 생성###
FN <- list.files(path = 'covid19daily') # 파일 이름
fileName<- unlist(str_extract_all(FN,"[0-9,-]*-2021"))

###파일 불러와서 일별로 계산후 데이터프레임에 합치기###
for(i in fileName){
  todayDF <- read.csv(paste0("covid19daily/",i,".csv"),header=T)
  todayDF <- todayDF %>% group_by(Country_Region) %>% summarise(Confirmed=sum(Confirmed), Deaths=sum(Deaths))
  dailyDF <- merge(yesterdayDF,todayDF, by ='Country_Region', all = TRUE)
  dailyDF <- dailyDF%>% group_by(Country_Region) %>% summarise(Confirmed=sub_(Confirmed.y,Confirmed.x), Deaths=sub_(Deaths.y,Deaths.x))
  date_month <- unlist(str_extract_all(i,"[0-9]{1,2}-[0-9]{1,2}"))
  names(dailyDF) <- c("Country_Region",paste0(date_month,"_확진자"),paste0(date_month,"_사망자"))
  resultDF <- merge(tempDF,dailyDF, by ='Country_Region', all = TRUE)
  yesterdayDF <- todayDF
  tempDF <- resultDF
}
rDF <- resultDF[,c(-2,-3)]


# write.csv(rDF, '일일데이터.csv')
#####음수값,결측값 갯수 세고 국가 체크##########
confirm <- rDF[,c(1,seq(2, ncol(rDF),2))]
death <- rDF[,c(1,seq(3, ncol(rDF),2))]


missC <- confirm 
missC <- apply(confirm[,c(2:ncol(confirm))]<0,1,sum,na.rm=T)
confirm[missC!=0,1]

missD <- death
missD <- apply(death[,c(2:ncol(death))]<0,1,sum,na.rm=T)
death[missD!=0,1]

nullC <- apply(is.na(confirm),1,sum)
nullD <- apply(is.na(death),1,sum)
confirm[nullC!=0,1]
death[nullD!=0,1]


####음수값, 결측치 0으로 대체 #######
rDF <- replace(rDF[,c(1,2:ncol(rDF))],rDF[,c(1,2:ncol(rDF))]<0,0)
rDF[is.na(rDF)] <- 0

rDF


# (1) 일별 국가별 코로나 발생자수와 사망자 수를 기준으로 전처리 하시오. 일부
# 국가는 지역별로 코로나 발생자수와 사망자 수가 분리되어 있으니 국가별로
# 집계하고 국가, 총발생자수, 총사망자수, 일평균 발생자수, 일평균 사망자수 리
# 스트를 제시하시오.(누적데이터인 경우 누적데이터로 해당 결과를 제시하고, 일별 데이터
# 를 산출하여 총합과 일평균값을 산출하여 결과 비교)

finalDF <- rDF
len <- ncol(finalDF)
finalDF$Confirmed <- apply(finalDF[,seq(2, len,2)],1,sum,na.rm=T)
finalDF$Deaths <- apply(finalDF[,seq(3, len,2)],1,sum,na.rm=T)
finalDF <- subset(finalDF,select=c(Country_Region,Confirmed,Deaths))
finalDF$MeanConfirmed <- avg_(finalDF$Confirmed,missC,nullC)
finalDF$MeanDeaths <- avg_(finalDF$Deaths,missD,nullD)
finalDF$missC <- missC
finalDF$missD <- missD
finalDF$nullC <- nullC
finalDF$nullD <- nullD
finalDF <- merge(finalDF,test_sub_DF, by ='Country_Region', all = TRUE)
print(finalDF)

# write.csv(finalDF, '최종DataFrame.csv')
# (2) 데이터가 0인 경우(코로나 환자 0)와 데이터가 없는 경우를 구분하여 전처
# 리하고 전처리 시 data가 없는 국가는 제외하고 제외된 국가 리스트를 제시하
# 시오
print("확진자 이상값 존재 국가 리스트");confirm[missC!=0,1]
print("사망자 이상값 존재 국가 리스트");death[missD!=0,1]
print("확진자 결측값 존재 국가 리스트");confirm[nullC!=0,1]
print("사망자 결측값 존재 국가 리스트");death[nullD!=0,1]



# (3) 2021년 1년동안 코로나 총 발생자수, 총 사망자수, 일평균 발생자수, 일평균
# 사망자 수를 기준으로 가장 많은 20개 국가를 내림차순으로 정렬하고 총 발생
# 자수, 총 사망자수, 일평균 발생자수, 일평균 사망자 수를 리포트 하시오. (4가
# 지 기준 각각 sorting)

totConfirmed20 <- finalDF[order(finalDF$Confirmed.x,finalDF$Deaths.x,
                                finalDF$MeanConfirmed,finalDF$MeanDeaths
                                ,decreasing = TRUE),]
rownames(totConfirmed20) <- 1 : length(rownames(totConfirmed20))
print(head(totConfirmed20,n=20))

meanConfirmed20 <- finalDF[order(finalDF$MeanConfirmed,finalDF$Confirmed.x,finalDF$Deaths.x,finalDF$MeanDeaths,decreasing = TRUE),]
rownames(meanConfirmed20) <- 1 : length(rownames(meanConfirmed20))
print(head(meanConfirmed20,n=20))

totDeaths20 <- finalDF[order(finalDF$Deaths.x,finalDF$Confirmed.x,finalDF$MeanConfirmed,finalDF$MeanDeaths,decreasing = TRUE),]
rownames(totDeaths20) <- 1 : length(rownames(totDeaths20))
print(head(totDeaths20,n=20))

meanDeaths20 <- finalDF[order(finalDF$MeanDeaths,finalDF$Confirmed.x,finalDF$Deaths.x,finalDF$MeanConfirmed,decreasing = TRUE),]
rownames(meanDeaths20) <- 1 : length(rownames(meanDeaths20))
print(head(meanDeaths20,n=20))

# write.csv(totConfirmed20, '확진자정렬.csv')
# write.csv(meanConfirmed20, '평균확진자정렬.csv')
# write.csv(totDeaths20, '사망자정렬.csv')
# write.csv(meanDeaths20, '평균사망자정렬.csv')

# (4) 2021년 1년동안 대한민국에서 발생한 총 코로나 발생자수와 총 사망자 수
# 와 일평균 발생자수와 일평균 사망자 수를 리포트 하시오.

KOR <- finalDF%>%subset(finalDF$Country=="Korea, South");KOR


# write.csv(KOR, '대한민국.csv')

#####################국가별 확진자 최대치인 달###################
maxconfirm_month <- apply(rDF,1,function(x){str_extract(names(which.max(x)),"[0-9]{2}")})
maxconfirm_month <- as.integer(maxconfirm_month);maxconfirm_month
hist(maxconfirm_month)
table(maxconfirm_month)
############Test용####################

test_sub_DF <- data.frame(test_sub_DF[,c(1,2,3)])
test_sum_DF <- subset(finalDF,select=c(Country_Region,Confirmed.x,Deaths.x))


#########단순 뺀것과 합계낸것 검증#############
###########정규성 검정###############

shapiro.test(test_sum_DF$Confirmed) ##p-value
shapiro.test(test_sub_DF$Confirmed)
hist(test_sum_DF$Confirmed, freq=FALSE, breaks=100) 
lines(density(test_sum_DF$Confirmed), col="blue", lwd=2)

hist(test_sub_DF$Confirmed, freq=FALSE, breaks=100) 
lines(density(test_sub_DF$Confirmed), col="red", lwd=2)

qqnorm(test_sum_DF$Confirmed, ylim = c(0,8e+05), xlim = c(-1, 1))
qqline(test_sum_DF$Confirmed,col="blue", lwd=3)

qqnorm(test_sub_DF$Confirmed, ylim = c(0,8.0e+05), xlim = c(-1, 1))
qqline(test_sub_DF$Confirmed,col="red", lwd=3)

#p-value가 2.2e-16로 0.05보다 작고,멱함수 분포를 따르고, Q-Q plot이 곡선형이기 때문에 정규성을 따르지 않는다.
#정규성을 따르지 않기 때문에 비모수검증 윌콕슨 순위합 검정(Wilcoxon rank sum test)을 이용하였다.

wilcox.test(test_sum_DF$Confirmed,test_sub_DF$Confirmed,alternative = 'g',conf.int=F,conf.level=0.975)
wilcox.test(test_sum_DF$Confirmed,test_sum_DF$Confirmed,alternative = 'g',conf.int=F,conf.level=0.975)
#p-value = 0.4951로서 유의수준 5%에서 귀무가설(H0:두 집단 간 차이는 없다)을 채택했다.

#####두 통계방식간 상관관계 분석#######
cor.test(test_sum_DF$Confirmed,test_sub_DF$Confirmed)
#95% 신뢰수준:[0.9999636, 0.9999792], 상관관계 0.99997로 강한 양의 상관관계

##################################################
##########확진자,사망자 상관관계######
cor.test(test_sum_DF$Confirmed.x,test_sum_DF$Deaths.x,method = 'pearson')
#95% 신뢰수준:[0.8778711 0.9283625], 상관관계 0.9062로 강한 양의 상관관계

##### 사망률 = (사망자/확진자)*100######
test_sum_DF$DeathRate <- round((test_sum_DF$Deaths.x/test_sum_DF$Confirmed.x)*100,2)
test_sum_DF <- test_sum_DF[order(test_sum_DF$DeathRate,test_sum_DF$Confirmed.x,test_sum_DF$Deaths.x,decreasing = TRUE),]
head(test_sum_DF,n=20)
# write.csv(test_sum_DF, '사망률.csv')
