install.packages("rJava")
install.packages("DBI")
install.packages("RJDBC")
install.packages("sqldf")
remove.packages("rJava", lib="~/R/win-library/4.0")

library(DBI)
library(rJava)
library(RJDBC)
library(sqldf)

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk1.8.0_202")


# Drive 설정
drv <- JDBC("oracle.jdbc.driver.OracleDriver", "C:/OracleTest/ojdbc6.jar")
# 데이터베이스 연결
conn <- dbConnect(drv,"jdbc:oracle:thin:@//127.0.0.1:1521/xe", "scott2", "tiger")
conn1 <- dbConnect(drv,"jdbc:oracle:thin:@//127.0.0.1:1521/xe", "scott3", "tiger")
# DB 레코드 검색, 추가, 수정, 삭제하기
# 모든 record
query = "SELECT * FROM test_table"
# query = "SELECT * FROM EMP"
dbGetQuery(conn, query)


# 나이 기준으로 내림차순 정렬
query = "SELECT * FROM test_table order by age desc"
dbGetQuery(conn, query)
# insert record
query = "insert into test_table values('kang', '1234', '강감찬', 45)"
dbSendUpdate(conn, query)
# 추가 확인
query = "SELECT * FROM test_table"
dbGetQuery(conn, query)
# 나이가 40세 이상인 record 
query = "select * from test_table where age >= 40
"
result <- dbGetQuery(conn, query)
result
# name이 '강감찬'인 데이터의 age를 40으로 수정
query = "update test_table set age = 40 where name = '강감찬'"
dbSendUpdate(conn, query)
# 수정된 레코드 조회
query = "select * from test_table where name = '강감찬'"
dbGetQuery(conn, query)

# name이 '홍길동'인 레코드 삭제
query = "delete from test_table where name = '홍길동'"
dbSendUpdate(conn, query)
query = "select * from test_table"
dbGetQuery(conn, query)

query ="insert into test_table values('kg', '4234', '김감찬', 45)"
query ="insert into test_table values('yim', '1324', '더좋은', 30)"
query ="insert into test_table values('tje', '2234', '더조은', 25)"
dbSendUpdate(conn, query)

query ="insert into test_table values('hm', '1334', '으은', 30);insert into test_table values('te', '2214', '호은', 25)"
dbSendUpdate(conn, query)

dbGetQuery(conn, query)


create sequence myboard_seq;

query = "create table myboard (
  num number(4) primary key,
  author varchar2(12),
  title varchar2(30),
  content varchar2(60)
)"
dbSendUpdate(conn, query)

query = "create sequence myboard_seq"
dbSendUpdate(conn, query)

query = "insert into myboard(num, author, title, content)
values(myboard_seq.nextval, '홍길동', '타이틀','내용')"
dbSendUpdate(conn, query)

query = "SELECT * FROM myboard"
dbGetQuery(conn, query)

query = "select e.empno, e.ename, e.sal, d.deptno, d.dname
from emp e, dept d
where e.deptno = d.deptno and e.job ='SALESMAN'"
dbGetQuery(conn, query)

query = "select e.ename, e.job, d.loc
from emp e, dept d
where e.deptno = d.deptno and e.ename like '%A%'"
dbGetQuery(conn, query)
#
#############################scott3 접속###########################
query = "SELECT * FROM EMP"
dbGetQuery(conn1, query)
query = "select max(sal) from emp
group by deptno
having max(sal) > 500"
dbGetQuery(conn1,query)

query="select ename, sal, deptno from emp
where sal > all (select sal from emp where deptno = 30)"
dbGetQuery(conn1,query)

###############################R에서 csv파일 db에 저장##############
##https://m.blog.naver.com/leebisu/222526411951

getwd()
sample <- read.csv(file="grade_csv.csv",sep=',')
dbWriteTable(conn1,"grade",sample)
query="select * from grade"
dbGetQuery(conn1,query)
query ="insert into grade values('1', '4', 40, 45,100)"
dbSendUpdate(conn1,query)
query="select * from grade"
dbGetQuery(conn1,query)
query ="delete from grade where ID='1'"
dbSendUpdate(conn1,query)
query="select * from grade"
dbGetQuery(conn1,query)
sqldf('select * from grade')

####################sqldf 사용해보기 ############
##https://cran.r-project.org/web/packages/sqldf/sqldf.pdf
sqldf.connection = (drv,"jdbc:oracle:thin:@//127.0.0.1:1521/xe", "scott3", "tiger")
sqldf('select * from grade', connection=getOption('scott3') )
sqldf()
getOption("sqldf.connection")

##################분산분석############
data <- read.csv("Part3/three_sample.csv", header=T)
head(data)
data
#2단계: 데이터 전처리 (NA, 이상치 제거)

data <- subset(data, !is.na(score), c(method, score))
head(data)
#3단계: 차트이용 outlier보기(데이터 분포 현황 분석)
par(mfrow = c(1, 2))
plot(data$score)
barplot(data$score)
mean(data$score)
#4단계: 데이터 정제(이상치 제거, 평균(14)이상 제거)
length(data$score)
data2 <- subset(data, score < 14)
length(data2$score)
#5단계: 정제된 데이터 확인
x <- data2$score
par(mfrow = c(1, 1))
boxplot(x)


#1단계: 세집단 subset 작성
data2$method2[data2$method == 1] <- "방법1"
data2$method2[data2$method == 2] <- "방법2"
data2$method2[data2$method == 3] <- "방법3"
#2단계: 교육 방법별 빈도수

table(data2$method2)
#3단계: 교육 방법을 x변수에 저장

x <- table(data2$method2)
x

#4단계: 교육 방법에 따른 시험성적 평균 구하기

y <- tapply(data2$score, data2$method2, mean)
y
tapply()함수
#https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/tapply
#https://www.guru99.com/r-apply-sapply-tapply.html

#5단계: 교육방법과 시험성적으로 데이터프레임 생성

df <- data.frame(교육방법 = x, 시험성적 = y)
df

# (3) 세 집단 간 동질성 검정
# barlett.test()함수 사용
# 검정 결과가 유의수준 0.05보다 큰 경우 세 집단 간 분포의 모양이 동질하다고 할 수
# 있다.
# 형식: barlett.test(종속변수 ~ 독립변수, data=dataset)
# 실습 (세 집단간 동질성 검정 수행)
bartlett.test(score ~ method, data = data2)
# 
# (4) 분산분석 (세 집단 간 평균 차이 검정)
# 세 집단 간의 동질성 검정에서 분포 형태가 동질하다고 분석되었기 때문에 aov()함수를
# 이용하여 세 집단 간 평균 차이 검정
# 동질하지 않다면 kruskal.test()함수 이용하여 비모수 검정을 수행
# 
# 실습 (분산분석 수행)
# R에서 ANOVA 분석을 수행하는 함수는 aov() 함수이다. 
# 참고로 R에는 anova() 함수도 있는데 이 함수는 회귀 모형 
# 등이 적합된 후 모형들 사이의 분산 분석을 수행할 때 이용된다. 
# aov() 함수는 첫번째 인수로 ANOVA 모형을 나타내는 수식을,
# 두번째 인수로 사용할 데이터를 입력받는다. 일원분류 분산 
# 분석은 반응변수를 나타내는 열이 y, 요인을 나타내는 열이 
# f라고 하면 다음 형식으로 수식이 입력된다.

# aov(y ~ f, data)

help(aov)
result <- aov(score ~ method2, data = data2)
names(result)
summary(result)
TukeyHSD(result)

result
plot(TukeyHSD(result))
