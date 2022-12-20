a<-5
b<-3
a

print(a,b)
dim(available.packages())
a<-available.packages()
head(a)
sessionInfo()
# 실습: stringr 패키지 설치
install.packages(("stringr"))
library(stringr)
require(stringr)
data()
hist(Nile)
hist(Nile, freq = F)
lines(density(Nile))
Nile 
getwd()
var1 <- 0
var1
var2 <- 0
var2
goods.code <- 'a001'
goods.name <- '냉장고'
goods.price <- 85000
goods.des <- '최고사양, 동급 최고 품질'
goods.name
age <- 35
names <- c('홍길동','이순신','유관순')
names
sum(10,20,20)
ls()
ls(3)
rm()
string <- '홍길동'
int <- 2
is.character(string)
x <- is.numeric(int)
x
is.logical(boolean)
is.logical(x)
is.na(x)

gender <- c("man", "woman", "woman", "man", "man")
Ngender <- as.factor(gender)
table(Ngender)
plot(Ngender)
class(Ngender)
mode(Ngender)
args(factor)
Ogender <- factor(gender, levels = c("woman", "man"), ordered = T)
Ogender
par(mfrow = c(2, 2))
plot(Ngender)
plot(Ogender)    

example(max)
example(seq)
mean(10:20)

# Chapter 02

# 실습: c() 함수를 이용한 벡터 객체 생성
c(1:20)
1:20
c(1, 2, 3, 4, 5)

# 실습: seq() 함수를 이용한 벡터 객체 생성
seq(1, 10, 2)

# rep() 함수를 이용한 벡터 생성
rep(1:3, 3)
rep(1:3, each = 3)

# 실습: union(), setdiff() 그리고 intersect() 함수를 이용한 벡터 자료 처리 
x <- c(1, 3, 5, 7)
y <- c(3, 5)
union(x, y)
setdiff(x, y)
intersect(x, y)


# 실습: 숫자형, 문자형 논리형 벡터 생성
v1 <- c(33, -5, 20:23, 12, -2:3)
v2 <- c("홍길동", "이순신", "유관순")
v3 <- c(T, TRUE, FALSE, T, TRUE, F, T)
v1; v2; v3


v4 <- c(33, 05, 20:23, 12, "4")
v4
IQR(v4)

v1; mode(v1); class(v1)
v2; mode(v2); class(v2)
v3; mode(v3); class(v3)

age <- c(30, 35, 40)
age
names(age) <- c("홍길동", "이순신", "강감찬")
age
age <- NULL

a <- c(1:50)
a[10:45]
a[19: (length(a) - 5)]

a[1, 2]

v1 <- c(13, -5, 20:23, 12, -2:3)
v1[1]
v1[c(2, 4)]
v1[c(3:5)]
v1[c(4, 5:8, 7)]



v1[-1]; v1[-c(2, 4)]; v1[-c(2:5)]; v1[-c(2, 5:10, 1)]


install.packages("RSADBE")
library(RSADBE)
data(Severity_Counts)
str(Severity_Counts)

Severity_Counts



m <- matrix(c(1:5))
m




m <- matrix(c(1:10), nrow = 2)
m



m <- matrix(c(1:11), nrow = 2)
m



m <- matrix(c(1:10), nrow = 2, byrow = T)
m


x1 <- c(m, 40, 50:52)
x2 <- c(30, 5, 6:8)
mr <- rbind(x1, x2)
mr

 
mc <- cbind(x1, x2)
mc


m3 <- matrix(10:19, 2)
m4 <- matrix(10:20, 2)
m3  
mode(m3); class(m3)  


m3[1, ]
m3[ , 5]
m3[2, 3]
m3[1, c(2:5)]


x <- matrix(c(1:9), nrow = 3, ncol = 3)
x1 <- matrix(c(1:11), nrow = 3, ncol = 3); x1 #매트릭스보다 데이터량이 많아서 데이터누락 발생
x


length(x)
length(x1)
ncol(x)


apply(x, 1, max)
apply(x, 1, min)
apply(x, 2, mean)



f <- function(x) {
  x * c(1, 2, 3)
}
result <- apply(x, 1, f)
result



result <- apply(x, 2, f)
result



colnames(x) <- c("one", "two", "three")
x

 
vec <- c(1:12)
arr <- array(vec, c(3, 2, 2))
arr

arr[ , , 1]
arr[ , , 2]
mode(arr); class(arr)



library(RSADBE)
data("Bug_Metrics_Software")
data()

str(Bug_Metrics_Software)

Bug_Metrics_Software


no <- c(1, 2, 3)
name <- c("hong", "lee", "kim")
pay <- c(150, 250, 300)
vemp <- data.frame(No = no, Name = name, Pay = pay)
vemp



m <- matrix(
  c(1, "hong", 150,
    2, "lee", 250,
    3, "kim", 300), 3, by = T)
memp <- data.frame(m)
memp


getwd()
txtemp <- read.table('part1/emp.txt', header = 1, sep = "")
txtemp


csvtemp <- read.csv('part1/emp.csv', header = T)
csvtemp
help(read.csv)
name <- c("사번", "이름", "급여")
read.csv('part1/emp2.csv', header = 1, col.names = name)

 
df <- data.frame(x = c(1:5), y = seq(2, 10, 2), z =seq(3, 15, 3))
df
plot(df)


# 실습: 데이터프레임의 칼럼명 참조하기 
df$y

# 실습: 데이터프레임의 자료구조, 열 수, 행 수, 칼럴명 보기
str(df)
ncol(df)
nrow(df)
names(df)
df[c(2:3), 1]

# 실습: 요약 통계량 보기 
summary(df)


df
# 실습: 데이터프레임 자료에 함수 적용하기 
apply(df[ , c(1, 2,3)], 2, sum)

# subset() 함수: data.frame()에서 조건에 만족하는 행을 추출하여 독립된 객체인 subset 생성
# 형식: 변수 <-subset(data.frame, 조건
# 실습: 데이터프레임의 부분 객체 만들기 
x1 <- subset(df, x >= 3)
x1

y1 <-subset(df, y<=8)
xyand <- subset(df, x<=2 & y <=6)
xyor <- subset(df, x<=2 | y <=0)

y1
xyand
xyor


# 실습: student 데이터프레임 만들기 
sid = c("A", "B", "C", "D")
score = c(90, 80, 70, 60)
subject = c("컴퓨터", "국어국문", "소프트웨어", "유아교육")

student <- data.frame(sid, score, subject)
student


# 실습: 자료형과 자료구조 보기 
mode(student); class(student)
str(sid); str(score); str(subject)
str(student)


# 실습: 두 개 이상의 데이터프레임 병합하기 
# 단계 1: 병합할 데이터프레임 생성
height <- data.frame(id = c(3,2,1), h = c(180, 175,176))
weight <- data.frame(id = c(2, 1,3), w = c(80, 75,56))

# 단계 2: 데이터프레임 병합하기 
user <- merge(height, weight, by.x = "id", by.y = "id")
user


# 
# 4.3 data.frame 객체 데이터 셋
#(galton 데이터 셋)


install.packages("UsingR")
library(UsingR)
data(galton)
as.vector(galton)
galton
help(galton)
str(galton)
dim(galton)
head(galton, 15)


# key를 생략한 list 생성하기
list <- list("lee", "이순신", 95)
list


# 리스트를 벡터 구조로 변경하기 
unlist <- unlist(list)
unlist

# 실습: 1개 이상의 값을 갖는 리스트 객체 생성하기 
num <- list(c(1:5), c(6, 10))
num


# key와 value 형식으로 리스트 객체 생성하기 
member <- list(name = c("홍길동", "유관순"), age = c(35, 25),
               address = c("한양", "충남"), gender = c("남자", "여자"),
               htype = c("아파트", "오피스텔"))
member

member$name
member$name[1]
member$name[2]

#key를 이용하여 value에 접근하기 
member$age[1] <- 45
member$id <- "hong"
member$pwd <- "1234"
member
member$age <- NULL
member
length(member)
mode(member); class(member)

#리스트 객체에 함수 적용하기 
a <- list(c(1:5))
b <- list(c(6:10))
lapply(c(a, b), max)


#리스트 형식을 벡터 형식으로 반환하기 
sapply(c(a, b), max) 


#다차원 리스트 객체 생성하기 
multi_list <- list(c1 = list(1, 2, 3),
                   c2 = list(10, 20, 30), 
                   c3 = list(100, 200, 300))
multi_list$c1; multi_list$c2; multi_list$c3
multi_list
#다차원 리스트를 열 단위로 바인딩하기 
do.call(cbind, multi_list)
class(do.call(cbind, multi_list))


#문자열 추출하기 
install.packages("stringr")
library(stringr)
str_extract("홍길동35이순신45유관순25", "[1-9]{2}")
str_extract_all("홍길동35이순신45유관순25", "[1-9]{2}")


#반복 수를 지정하여 영문자 추출하기 
string <- "hongkd105leess1002you25강감찬2005"
str_extract_all(string, "[a-z]{3}")
str_extract_all(string, "[a-z]{3,}")
str_extract_all(string, "[a-z]{3,5}")


#문자열에서 한글, 영문자, 숫자 추출하기 
str_extract_all(string, "hong")
str_extract_all(string, "25")
str_extract_all(string, "[가-힣]{3}")
str_extract_all(string, "[a-z]{3}")
str_extract_all(string, "[0-9]{4}")

#문자열에서 한글, 영문자, 숫자를 제외한 나머지 추출하기 
str_extract_all(string, "[^a-z]")
str_extract_all(string, "[^a-z]{4}")
str_extract_all(string, "[^가-힣]{5}")
str_extract_all(string, "[^0-9]{3}")


# 주민등록번호 검사하기 
jumin <- "123456-1234567"
str_extract(jumin, "[0-9]{6}-[1234][0-9]{6}")
str_extract_all(jumin, "\\d{6}-[1234]\\d{6}")


#지정된 길이의 단어 추출하기 
name <- "홍길동1234,이순신5678,강감찬1012"
str_extract_all(name, "\\w{7,}")


#문자열의 길이 구하기 
string <- "hongkd105leess1002you25강감찬2005"
len <- str_length(string)
len

#문자열 내에서 특정 문자열의 위치(index) 구하기 
string <- "hongkd105leess1002you25강감찬2005"
str_locate(string, "강감찬")


#부분 문자열 만들기 
string_sub <- str_sub(string, 1, len - 7)
string_sub
string_sub <- str_sub(string, 1, 23)
string_sub


#대문자, 소문자 변경하기 
ustr <- str_to_upper(string_sub); ustr
str_to_lower(ustr)

#문자열 교체하기
string_sub
string_rep <- str_replace(string_sub, "hongkd105", "홍길동35,")
string_rep <- str_replace(string_rep, "leess1002", "이순신45,")
string_rep <- str_replace(string_rep, "you25", "유관순25,")
string_rep


# 문자열 결합하기 
string_rep
string_c <- str_c(string_rep, "강감찬55")
string_c

# 문자열 분리하기 
string_c
string_sp <- str_split(string_c, ",")
string_sp


# 문자열 합치기
# 문자열 벡터 만들기 
string_vec <- c("홍길동35", "이순신45", "유관순25", "강감찬55")
string_vec

# 콤마를 기준으로 문자열 벡터 합치기 
string_join <- paste(string_vec, collapse = ",")
string_join


rm(list=ls())

edit(df)
df=data.frame()
edit(df)
df

score <- scan()

if(score >= 90) {
  result = "A학점"
} else if(score >= 80) {
  result = "B학점"
} else if(score >= 70) {
  result = "C학점"
} else if(score >= 60) {
  result = "D학점"
} else {
  result = "F학점"
}
cat("당신의 학점은", result)
print(result)

scan("")
num <- scan("")
num
sum(num)
# Chapter 04

# 실습: 산술연산자 사용
num1 <- 100
num2 <- 20
result <- num1 + num2
result
result <- num1 - num2
result
result <- num1 * num2
result
result <- num1 / num2
result

result <- num1 %% num2
result

result <- num1 ^ 2
result
result <- num1 ^ num2
result

#  관계연산자 사용
boolean <- num1 == num2
boolean
boolean <- num1 != num2
boolean

boolean <- num1 > num2
boolean
boolean <- num1 >= num2
boolean
boolean <- num1 < num2
boolean
boolean <- num1 <= num2
boolean

#  논리연
logical <- num1 >= 50 & num2 <= 10
logical
logical <- num1 >= 50 | num2 <= 10
logical

logical <- num1 >= 50
logical

logical <= !(num1 >= 50)
logical

x <- TRUE; y <- FALSE
xor(x, y)


# 실습: if() 함수 사용하기 
x <- 50; y <- 4; z <- x * y
if(x * y >= 40) {
  cat("x * y의 결과는 40이상입니다.\n")
  cat("x * y = ", z)
} else {
  cat("x * y의 결과는 40미만입니다. x * y = ", z, "\n")
}

# 실습: if() 함수 사용으로 입력된 점수의 학점 구하기 
score <- scan()

score
result <- "노력"
if(score >= 80) {
  result <- "우수"
}
cat("당신의 학점은 ", result, score)


# 실습: if~else if 형식으로 학점 구하기 
score <- scan()
if(score >= 90) {
  result = "A학점"
} else if(score >= 80) {
  result = "B학점"
} else if(score >= 70) {
  result = "C학점"
} else if(score >= 60) {
  result = "D학점"
} else {
  result = "F학점"
}
cat("당신의 학점은", result)
print(result)


#  ifelse() 함수 사용하기 
score <- scan()

ifelse(score >= 80, "우수", "노력")
ifelse(score <= 80, "우수", "노력")

#  ifelse() 함수 응용하기 
excel <- read.csv("C:/Rwork/Part-I/excel.csv", header = T)
q1 <- excel$q1
q1
ifelse(q1 >= 3, sqrt(q1), q1)

#  ifelse() 함수에서 논리연산자 사용하기 
ifelse(q1 >= 2 & q1 <= 4, q1 ^ 2, q1)

switch("name", id = "hong", pwd = "1234", age = 105, name = "홍길동")

#switch() 함수를 사용하여 사원명으로 급여저보 보기 
empname <- scan(what = "")
empname
switch(empname, 
       hong = 250, 
       lee = 350,
       kim = 200,
       kang = 400
)

# 벡터에서 which() 함수 사용: index 값을 반환
name <- c("kim", "lee", "choi", "park")
which(name == "choi")


#  데이터프레임에서 which() 함수 사용
#  벡터 생성과 데이터프레임 생성
no <- c(1:5)
name <- c("홍길동", "이순신", "강감찬", "유관순", "김유신")
score <- c(85, 78, 89, 90, 74)
exam <- data.frame(학번 = no, 이름 = name, 성적 = score)
exam

# 일치하는 이름의 위치(인덱스) 반환
which(exam$이름 == "유관순")
exam[4, ]


#  for() 함수 사용 기본
i <- c(1:10)
for(n in i) {
  print(n * 10)
  print(n)
}

# 짝수 값만 출력하기 
i <- c(1:10)
for(n in i)
  if(n %% 2 == 0) print(n)

# 짝수이면 넘기고, 홀수 값만 출력하기 
i <- c(1:10)
for(n in i) {
  if(n %% 2 == 0) {
    next
  } else {
    print(n)
  }
}


#  변수의 칼럼명 출력하기 
name <- c(names(exam))
for(n in name) {
  print(n)
}

# 벡터 데이터 사용하기 
score <- c(85, 95, 98)
name <- c("홍길동", "이순신", "강감찬")

i <- 1
for(s in score) {
  cat(name[i], " -> ", s, "\n")
  i <- i + 1
}


# while() 함수 사용하기 
i = 0
while(i < 10) {
  i <- i + 1
  print(i)
}


# 매개변수가 없는 사용자 함수 정의하기 
f1 <- function() {
  cat("매개변수가 없는 함수")
}

f1()

# 결과를 반환하는 사용자 함수 정의하기 
f3 <- function(x, y) {
  add <- x + y
  return(add)
}

add <- f3(10, 20)
add

x <- c(7,5,12,9,15,6)
var(x)
sd(x)

# 4.2 기술통계량을 계산하는 함수 정의
# 실습 (기본함수를 이용하여 요약통계량과 빈도수 구하기


test <- read.csv("test.csv", header = TRUE)
head(test)
#summary()함수, table()함수
# 요약 통계량 구하기
summary(test)
# 특정 변수의 빈도수 구하기
table(test$A)
# 각 칼럼 단위의 빈도수와 최대값, 최소값 계산을 위한 사용자 함수 정의하기
data_pro <- function(x) {
  for(idx in 1:length(x)) {
    cat(idx, "번째 칼럼의 빈도 분석 결과")
    print(table(x[idx]))
    cat("\n")
  }
  
  for(idx in 1:length(x)) {
    f <- table(x[idx])
    cat(idx, "번째 칼럼의 최대값/최소값\n")
    cat("max = ", max(f), "min = ", min(f), "\n")
  }
}
data_pro(test)

pytha <- function(s, t) {
  a <- s ^ 2 - t ^ 2
  b <- 2 * s * t
  c <- s ^ 2 + t ^ 2
  cat("피타고라스 정리: 3개의 변수: ", a, b, c)
}
pytha(2, 1)

# 결측치를 포함하는 자료를 대상으로 평균 구하기 
# 결측치(NA)를 포함하는 데이터 생성
data <- c(10, 20, 5, 4, 40, 7, NA, 6, 3, NA, 2, NA)

# 결측치 데이터를 처리하는 함수 정의 
na <- function(x) {
  #  NA 제거 
  print(x)
  print(mean(x, na.rm = T))
  
  #  NA를 0으로 대체 
  data = ifelse(!is.na(x), x, 0)
  print(data)
  print(mean(data))
  
  # NA를 평균으로 대체 
  data2 = ifelse(!is.na(x), x, round(mean(x, na.rm = TRUE), 2))
  print(data2)
  print(mean(data2))
}

# 결측치 처리를 위한 사용자 함수 호출
na(data)

seq(-2, 2, by = .2)

install.packages("dplyr")
library(dplyr)
iris %>% head()
iris %>% head() %>% subset(Sepal.Length >= 5.0)
library(dplyr)
# 1:5의 인자를 다시 쓰지 않고 사용할 수 있으며 sum 함수로 호출한다.
1:5 %>% sum(.)
# length 함수에 의해서 길이인 5까지 더해져서 20을 반환한다.
1:5 %>% sum(length(.))
tbl_df()

installed.packages() %>%
  as.data.frame() %>%
  select_("Package"," Version") %>%
  filter(Package %in% c("dplyr", "tidyr"))

install.packages(c("dplyr", "hflights"))
install.packages("magrittr")


# Chapter 06

# 실습: iris 데이터 셋을 대상으로 '%>%' 기호를 이용하여 함수 적용하기 
install.packages("dplyr")
library(dplyr)
iris
iris %>% head()

iris %>% head() %>% subset(Sepal.Length >= 5.0)


# 실습: dplyr 패키지와 hflight 데이터 셋 설치 
install.packages(c("dplyr", "hflights"))
library(dplyr)
library(hflights)

str(hflights)

# 1:5의 인자를 다시 쓰지 않고 사용할 수 있으며 sum 함수로 호출한다.
1:5 %>% sum(.)
# length 함수에 의해서 길이인 5까지 더해져서 20을 반환한다.
1:5 %>% sum(length(.))
# sum(length(1:5))와 같이 수열을 지정하여 반환하므로 수열의 길이인 5까지 더해서
# 20을 반환한다.
5 %>% sum(1:.)
# { }(브레이스)를 적용하면 함수를 한 번씩만 실행한다.
5 %>% {sum(1:.)}
csvgrade <- read.csv("grade_csv.csv")
# 1부터 행의 수만큼 수열로 출력하고 나눈 나머지가 0인 행의 부분집합을 추출한다.
csvgrade %>% subset(1:nrow(.) %% 2 == 0)

.libPaths()
library(tidyr)

library(dplyr)
csvgrade <- read.csv("grade_csv.csv")
csvgrade %>% filter(class == 1)
csvgrade %>% filter(class != 1)
csvgrade %>% filter(math > 50)
csvgrade %>% filter(math < 50)

csvgrade %>% filter(eng >= 80)
csvgrade %>% filter(class == 1 & math >= 50)
csvgrade %>% filter(class %in% c(1, 3, 5))
class1 <- csvgrade %>% filter(class == 1)
mean(class1$math)
filter(hflights_df, Month == 1 | Month == 2) 


# 1.5 컬럼으로 데이터 검색
# 데이터 셋의 특정 컬럼을 기준으로 데이터 검색 시 select()함수 사용
# 형식: select(dataframe, 컬럼1, 컬럼2, …)
# select 함수는 추출하고자 하는 객체를 할당하면 해당 객체만 추출한다.
# 단일 객체
# 단일 객체를 추출하여 반환한다.
# Ex.
library(dplyr)
csvgrade <- read.csv("grade_csv.csv")
csvgrade %>% select(math)

csvgrade %>% select(math,class)
csvgrade %>% select(-math)


# %>% 함수 적용
# %>% 함수를 적용하여 함수를 조합하여 구현한다.
# %>% 함수를 적용하면 코드의 길이가 줄어들어 이해하기도 쉽고 실행하는데 불필요한
# 부분이 줄어들어 시간이 단축된다.
# 특정 객체의 값을 추출한다.
# Ex.
library(dplyr)
csvgrade <- read.csv("grade_csv.csv")
csvgrade %>% filter(class == 1 ) %>% select(eng)
csvgrade

# 특정 객체의 값 일부를 추출한다.

library(dplyr)
csvgrade <- read.csv("grade_csv.csv")
csvgrade %>% select(id, math) %>% head(3)


# 실습 (hflights_df를 대상으로 지정된 컬럼 데이터 검색)
select(hflights_df, Year, Month, DepTime, ArrTime)
hflights_df %>% select(hflights_df, Year, Month, DepTime, AirTime)

# 실습 (hflights_df 대상 컬럼의 범위로 검색)
select(hflights_df, Year:ArrTime)
# select()함수 사용시 특정 컬럼만이 아닌 컬럼의 범위 설정 가능
# 검색조건으로 시작걸럼:종료컬럼 형식으로 컬럼 범위의 시작과 끝을 지정
# 특정 컬럼 또는 컬럼의 범위를 검색에서 제외하려는 경우 제외하려는 컬럼 이름 또는
# 범위 앞에 “-“속성을 지정
# 예) Year부터 DepTime컬럼까지 제외한 나머지 컬럼만 선택하여 검색할 때
# select(hflights_df, -(Year:DepTime)) 형식


# 1.6 데이터셋에 컬럼 추가
# 데이터 셋에 특정 컬럼을 추가하는 mutate()함수
# 형식: mutate(dataframe, 컬럼명1=수식1, 컬럼명2=수식2, …)
# 실습 (hflights_df에서 출발 지연시간과 도착 지연시간의 차이를 계산한 컬럼 추가)
mutate(hflights_df, gain = ArrTime - DepTime,
       gain_per_hour = gain / (AirTime / 60))
hflights_df %>% mutate(gain = (ArrDelay - DepDelay), gain_per_hour = gain / (AirTime /
                                                                             60))
# 실습 (mutate()함수에 의해 추가된 컬럼 보기)
select(mutate(hflights_df, gain = ArrDelay - DepDelay,
              gain_per_hour = gain / (AirTime / 60)),
       Year, Month, ArrDelay, DepDelay, gain, gain_per_hour)
# console창 크기 이외의 컬럼명 확인이 어려운 경우 select()함수 안에 mutate()함수 사용

# 1.7 요약통계 구하기
# 전체의 평균, 표준편차, 사분위수 등 전체적인 값들에 대한 요약 통계량을 산출할 때는
# summary 함수를 사용하고, 개별 컬럼의 데이터에 대한 요약 통계량을 구할 때는
# summarise 함수를 사용한다.
# 컬럼 데이터의 요약 통계량을 반환한다.
csvgrade <- read.csv("grade_csv.csv")
csvgrade %>% summarise(mean_math = mean(math))
csvgrade
summarise(csvgrade,cnt=n(),tot=mean(math,na.rm=TRUE))
summarise(hflights_df, avgAirTime = mean(AirTime, na.rm = TRUE))
# hflights_df %>% summarise(avgAirTime = mean(AirTime, na.rm = TRUE))
# mean()함수를 사용하여 평균을 계산하여 avgAirTime변수에 저장
# hflights_df %>% summarise(avgAirTime = mean(AirTime, na.rm=TRUE)) 형식 사용

summarise(hflights_df, cnt = n(),
          delay = mean(AirTime, na.rm = TRUE))
# n()함수: 데이터 셋의 관측치 길이를 구하는 함수

# 실습 (도착시간(AirTime)의 표준편차와 분산 계산)
summarise(hflights_df, arrTimeSd = sd(ArrTime, na.rm = TRUE),
          arrTimeVar = var(ArrTime, na.rm = T))

# 1.8 집단변수 대상 그룹화
library(dplyr)
csvgrade <- read.csv("grade_csv.csv")
csvgrade %>% group_by(class) %>% summarise(mean_math = mean(math))
csvgrade
csvgrade %>% group_by(class) %>% summarise(mean_math = mean(math), sum_math
                                           = sum(math), median_math = median(math))
species <- group_by(iris, Species)
str(species)
species
# species <- iris %>% group_by(species) 

# 1.10 데이터프레임 합치기
# 서로 다른 데이터프레임을 대상으로 행 단위 또는 열 단위로 합치는 함수
# [표 6.3] bind관련 함수
# bind_rows(df1, df2)
# bind_cols(df1, df2)
# 세로 결합
# dplyr 패키지의 bind_rows 함수로 나뉘어져 있는 데이터를 세로로 결합할 수 있다.
# 형식: bind_rows(dataframe1, dataframe2)
# Ex.
library(dplyr)
a <- data.frame(id = c(1, 2, 3, 4, 5), score = c(60, 80, 70, 90, 85));a
b <- data.frame(id = c(3, 4, 5, 6, 7), weight = c(80, 90, 85, 60, 85));b
# 세로로 결합한다.
bind_rows(a, b)
bind_cols(a,b)

# rbind 함수로 행을 결합하기 위해서는 Data Frame의 열 개수, 칼럼 이름이 같아야 한다
# 형식: rbind(a, b)
a <- data.frame(id = c(1, 2, 3, 4, 5), score = c(60, 80, 70, 90, 85));a
b <- data.frame(id = c(6, 7 , 8), score = c(80, 90, 85));b

rbind(a, b)

# 가로 결합
# dplyr 패키지의 bind_cols 함수로 나뉘어져 있는 데이터를 가로로 결합할 수 있다.
# 형식: bind_cols(dataframe1, dataframe2)
# 실습 (두 개의 데이터프레임을 열 단위로 합치기)
# df_cols <- bind_cols(df1, df2)
# df_cols
# cbind 함수로 열을 결합하여 반환한다.
# cbind 함수로 열을 결합하기 위해서는 Data Frame의 행 개수가 서로 같아야 한다.
# 형식: cbind(a,b)
a <- data.frame(id = c(1, 2, 3, 4, 5), score = c(60, 80, 70, 90, 85));a

b <- data.frame(age = c(20, 19 , 20, 19, 21), weight = c(80, 90, 85, 60, 85));b
cbind(a,b)

a <- data.frame(id = c(1, 2, 3, 4, 5), score = c(60, 80, 70, 90, 85))
a
b <- data.frame(id = c(3, 4 , 5, 6, 7), weight = c(80, 90, 85, 60, 85))
b
cbind(a, b)
merge(a,b,by="id",all=TRUE,na.rm=TRUE)

# 1.11 컬럼명 수정하기
df <- data.frame(one = c(4, 3, 8))
df
df <- rename(df, "원" = one)
df
# 실습 (데이터프레임의 컬럼명 수정)
df_rename <-rename(df_cols, x2 = x1)
df_rename <- rename(df_rename, y2 = y1)
df_rename

install.packages("reshape2")

data("Orange")
summary(Orange)
Orange
data(iris)
iris
library(dplyr)
iris%>%group_by(Species,Sepal.Width)%>%summarise(sum)

c<-c(15,83,45,91,8,67,48)
sd(c)
median(c)
mean(c)
std(c)
var(c)
quantile(c,3/4)
d<-c(2,3,3,4,4,4,4,5,5,6,7)
quantile(d,1/4)
quantile(c)
a<-c(18,10,14,8,20,6,14,16,20)
mean(a)

library(dplyr)
df<-read.csv('basic1.csv')
,header = T, stringsAsFactors=T,fileEncoding="UTF-8",encoding='CP949')
str(df)
df%>%arrange(-f4)#%>%arrange(f5)

library(mlbench)
install.packages("c:\\caret_6.0-90.tar.gz",repos=NULL,type="source")
