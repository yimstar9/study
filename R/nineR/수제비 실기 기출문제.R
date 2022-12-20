#수제비 빅분기 실기 기출문제 1유형
#1. BostonHousing 데이터세트
#crim항목의 상위에서 10번째값(즉, 상위10개의 값 중에서 가장 적은 값)으로 10개의 값을 변환하고 age가 80이상인 값에 대하여 crim의 평균을 구하시오
library(mlbench)
library(dplyr)
data("BostonHousing")
BostonHousing
df<-BostonHousing
ans<-df%>%arrange(-crim)%>%mutate(pre_crim=ifelse(crim>crim[10],crim[10],crim))%>%filter(age>=80)%>%summarise(mean=mean(pre_crim))
print(ans$mean)

#2. 주어진 데이터의 첫번째 행부터 순서대로 80%까지의 데이터를 훈련데이터로 추출후 'total_bedrooms'변수의 결측값(NA)을 'total_bedrooms'변수의 중앙값으로 대체하고
#대체 전의 'total_bedrooms'변수 표준편차 값과 대체후의 'total_bedrooms'변수 표준편차 값의 차이의 절대값을 구하시오
#http://www.kaggle.com/camnugent/california-housing-prices
df2<-read.csv("수제비데이터/housing.csv")
df2<-head(df2,nrow(df)*0.8)
ans2<-df2%>%mutate(pre_totalbedrooms=ifelse(is.na(total_bedrooms),median(total_bedrooms,na.rm=T),total_bedrooms))%>%summarise(ans=abs(sd(pre_totalbedrooms)-sd(total_bedrooms,na.rm=T)))
print(ans2$ans)

#3. 다음은 insurance 데이터 세트이다. charges항목에서 이상값의 합을 구하시오(이상값은 평균에서 1.5 표준편차 이상인 값)
#http://www.kaggle.com/mirichoi0218/insurance/version/1
df3<-read.csv("수제비데이터/insurance.csv")
head(df3)
low<-mean(df3$charges,na.rm=T)-sd(df3$charges,na.rm=T)*1.5
high<-mean(df3$charges,na.rm=T)+sd(df3$charges,na.rm=T)*1.5
ans3<-df3%>%filter(charges<low|charges>high)%>%summarise(sum=sum(charges))
print(ans3$sum)


#4. 주어진 housing 데이터 세트에서 결측값이 있는 모든 행을제거한 후 데이터의 순서대로 상위 70%의 데이터를 훈련데이터로 만들고 ,
#훈련데이터의 housing_median_age컬럼의 제1사분위수(Q1)를 정수로 구하시오
#http://www.kaggle.com/camnugent/california-housing-prices
df<-read.csv("수제비데이터/housing.csv")
summary(df)
df<-na.omit(df)
df_train<-head(df,nrow(df)*0.7)%>%summarise(quantile(housing_median_age, 0.25))
cat(as.integer(df_train))


#5. 다음은 타이타닉 데이터 세트이다. 데이터가 없는 것을 결측값으로 하여 결측값 비율을 구하고 결측값 비율이 가장
# 높은 컬럼의 이름을 구하시오
df<-read.csv("수제비데이터/titanic_train.csv")
df<-colSums(is.na(df))/nrow(df)
cat(names(which.max(df)))


#6. 다음은 연도별 각 나라의 결핵 감염에 대한 유병률의 데이터이다. 2000년도 국가의 평균 결핵 발생 건수를 구하고 
#2000년도의 결핵 발생 건수가2000년의 결핵 발생 건수보다 높은 국가의 수를 구하시오.
df<-read.csv("수제비데이터/TB_notifications_2022-11-22.csv")
ans<-df%>%filter(year==2000)%>%filter(new_sp>mean(new_sp,na.rm=T))%>%nrow
cat(ans)

#7. 다음과 같이 고양이 cats의 데이터 세트가 주어질 경우에 심장의 무게(Hwt)의 이상ㄱ밧의 평균을 구하시오.
#(단 MASS 패키지의 cats데이터 세트를 사용하고, 이상값은 평균에서 1.5배 표준편차를 벗어나는 값으로 한다.)
#(cats 데이터 세트는 MASS라이브러리에 있음)
data(cats)
df<-cats
hi_out <- mean(df$Hwt,na.rm=T)+1.5*sd(df$Hwt)
lo_out <- mean(df$Hwt,na.rm=T)-1.5*sd(df$Hwt)
ans<-df%>%filter(Hwt<lo_out|Hwt>hi_out)%>%summarise(mean=mean(Hwt))
cat(ans$mean)

#8. 다음은 23회의 우주왕복 임무에서 수집된 데이터이다. damage가 1 이상일 경우의 temp와 damage의
#피어슨 상관계수를 구하시오(faraway패키지의 orings 데이터 세트)
library(faraway)
data(orings)
df<-orings
ans<-df%>%filter(damage>=1)%>%cor
cat(ans[2])

#9. 주어진 데이터 세트는 32개 자동차 모델의 디잔이과 선응을 비교한 mtcars내장 데이터 세트이다.
#수동(am=1)중에서 가장 무게(wt)가 작게 나가는 10개 데이터의 평균 연비(mpg)와 자동(am=0)중에서
#가장 무게(wt)가 작게 나가는 10개 데이터의 평균 연비(mpg)의 차이를 구하시오.
data(mtcars)
df<-mtcars
str(mtcars)
am1<-df%>%filter(am==1)%>%arrange(wt)%>%slice(1:10)%>%summarise(mean=mean(mpg,na.rm=T))
am2<-df%>%filter(am==0)%>%arrange(wt)%>%slice(1:10)%>%summarise(mean=mean(mpg,na.rm=T))
ans<-am1$mean-am2$mean
cat(ans)

#10. 주어진 diamonds의 데이터를 순서대로 80% 데이터를 제거한후 cut이 'Fair'이면서 carat이 1이상인 
#diamonds 의 price의 최대값을 구하시오
data(diamonds)
df<-diamonds[-c(1:(nrow(diamonds)*0.8)),]
dim(df)
str(df)
ans<-df%>%filter(cut=='Fair'&carat>=1)%>%summarise(m=max(price))
cat(ans$m)

#11. 다음은 airquality데이터 세트이다. 8월 20일의 ozone 값을 구하시오
data("airquality")
df<-airquality
ans<-df%>%filter(Month==8&Day==20)%>%select(1)
cat(ans$Ozone)

#12. 다음은 iris 데이터 세트이다. Sepal.Length의 mean값과 Sepal.Width의 mean값의 합계를 구하시오
head(iris)
df<-iris
ans<-df%>%summarise(Lm=mean(Sepal.Length),Wm=mean(Sepal.Width))%>%apply(1,sum)
cat(ans[1])

#13. 다음은 mtcars 데이터 세트이다. 4기통(cyl)인 자동차의 비율을 구하시오
t<-nrow(mtcars)
cyl_4<-nrow(mtcars[mtcars$cyl==4,])
ans<-cyl_4/t
cat(ans)

#14. 다음은 mtcars 데이터 세트이다. 변속기어(gear) 수가 4이고 수동(am==1) 변속기인 데이터에서 자동차 
#연비(mpg)의 mean 값과 전체 마력(hp)의 표준편차의 합계를 구하시오
data(mtcars)
df<-mtcars
ans<-df%>%filter(gear==4&am==1)%>%summarise(m=mean(mpg),s=sd(hp))%>%apply(1,sum)
cat(ans[1])

#15. 다음은 BostonHousing 데이터 세트이다. crim 항목이 1보다 작거나 같은 경우에 medv 항목의 mean값을 구하시오
data("BostonHousing")
df<-BostonHousing
ans<-df%>%filter(crim<=1)%>%summarise(m=mean(medv))
cat(ans$m)

#16. 다음은 iris 데이터 세트이다. iris 데이터 세트에서 Species가 virginica인 항목에서 Sepal.Length가
#6보다 크면1, 아니면 0으로 파생컬럼 Len을 생성 후 Len컬럼의 Sum값을 구하시오
data(iris)
df<-iris
ans<-df%>%filter(Species=='virginica')%>%mutate(Len=ifelse(Sepal.Length>6,1,0))%>%summarise(s=sum(Len))
cat(ans$s)

#17. 다음은 airquality 데이터 세트이다. Ozone의 결측값을 평균 값으로 대체하고, median값에서 2*IQR 을 뺀
#값과 median값에서 2*IQR을 더한값 사이에 존재하는 Ozone값의 합계를 구하시오
data(airquality)
df<-airquality
df
ans<-df%>%mutate(pre_Ozone=ifelse(is.na(Ozone),mean(Ozone,na.rm=T),Ozone))%>%
      filter(pre_Ozone>=median(pre_Ozone)-2*IQR(pre_Ozone)&pre_Ozone<=median(pre_Ozone)+2*IQR(pre_Ozone))%>%
      summarise(s=sum(pre_Ozone))
cat(ans$s)

