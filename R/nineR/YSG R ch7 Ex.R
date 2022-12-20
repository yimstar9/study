#임성구

#1번. 본문에서 생성된 dataset2의 직급(postion)컬럼을 
# 대상으로 1급 -> 5급, 5급 -> 1급 형식으로 역코딩하여
# position2컬럼에 추가하시오.
dataset2$position2 <- 6-dataset2$position
dataset2$position2[dataset2$position2==1] <- '1급'
dataset2$position2[dataset2$position2==2] <- '2급'
dataset2$position2[dataset2$position2==3] <- '3급'
dataset2$position2[dataset2$position2==4] <- '4급'
dataset2$position2[dataset2$position2==5] <- '5급'
dataset2$position;dataset2$position2

#2번. 본문에서 생성된 dataset2의 resident 컬럼을 대상으로
# NA값을 제거한 후 resident2변수에 저장하시오.
dataset2$resident
resident2 <- subset(dataset2, !is.na(dataset2$resident));resident2


#3번. 본문에서 생성된 dataset2의 gender컬럼을 대상으로 
# 1 -> “남자”, 2-> “여자”로 코딩 변경하여 gender2 컬럼에 
# 추가하고, 파이차트로 결과를 확인하시오
dataset2$gender
dataset2$gender2[dataset2$gender==1]  <- '남자'
dataset2$gender2[dataset2$gender==2] <- '여자'
dataset2$gender2
pie(table(dataset2$gender2))

#4번. 본문에서 생성된 dataset2의 age컬럼을 대상으로 
# 30세이하 -> 1, 30-55세 -> 2, 55이상-> 3으로 리코딩하여 
# age3컬럼에 추가한 뒤에 age, age2, age3 컬럼만 확인하시오
dataset2$age
dataset2$age3[dataset2$age <= 30] <-1
dataset2$age3[dataset2$age > 30 & dataset2$age < 55] <-2
dataset2$age3[dataset2$age >= 55] <-3
dataset2[c('age', 'age3')];
dataset2$age2

#5번. 정제된 data를 대상으로 작업 디렉터리(“C/Rwork/”)에
# 파일 이름을 “cleandata.csv”로하여 따옴표와 행 이름을 
# 제거하여 저장하고, 저장된 파일의 내용을 읽어 new_data변수
# 에 저장하고 확인하시오.
setwd("c:/Rwork")
write.csv(dataset2,"cleandata.csv",quote=F,row.names = F)
new_data<-read.csv("cleandata.csv",header = T)
new_data
str(new_data)
#6번. dataset#3 내 “user_data.csv”, “return_data.csv”파일을
# 이용하여 고객별 반폼사유코드 (return_code)를 대상으로 
# 다음과 같이 파생변수를 추가하시오.
userdata<-read.csv("dataset#3/user_data.csv",header = T)
returndata <- read.csv("dataset#3/return_data.csv",header = T)
userdata
returndata
library(reshape2)
z <- dcast(returndata, user_id ~ return_code, length)
names(z)<-c('user_id','return_code1','return_code2','return_code3', 
                   'return_code4')
z
library(plyr)
userdata <- join(userdata, z, by='user_id')


#7. iris데이터를 이용하여 5겹 2회 반복하는 
# 교차 검정 데이터를 샘플링하시오.
library(cvTools)
cross <- cvFolds(nrow(iris),K=5,R=2)
str(cross)
table(cross$which)
cross

K=1:5
R=1:2

for(r in R){
    cat('R=',r,'\n')
    for(k in K) {
        datas_idx <- cross$subsets[cross$which == k, r]
        cat('K = ', k, '검정데이터 \n')
        print(iris[datas_idx, ])
        cat('K = ', k, '훈련데이터 \n')
        print(iris[-datas_idx, ])
    }
}

