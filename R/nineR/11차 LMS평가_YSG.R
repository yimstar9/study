#임성구


# 2. Oracle XE DB에 접근하기 위한 설정(driver 설정, 
# import package 또는 package loading, DB연결 등)을 하시오.

#import package
library(DBI)
library(rJava)
library(RJDBC)
#자바 환경변수 설정
Sys.setenv(JAVA_HOME = "C:\\Program Files\\Java\\jdk1.8.0_202")
# Driver 설정
drv <- JDBC("oracle.jdbc.driver.OracleDriver", "C:/OracleTest/ojdbc6.jar")
# 데이터베이스 연결
conn <- dbConnect(drv,"jdbc:oracle:thin:@//127.0.0.1:1521/xe", "scott", "tiger")


# 3.exam_table 내 모든 데이터를 조회하시오.
query = "SELECT * FROM EXAM_TABLE"
dbGetQuery(conn, query)

# ID PASS NAME SCORE
# 1 1001 1234  Kim    90
# 2 1002 3456  Lee   100
# 3 1003 5678 Park    85
# 4 1004 7890 Choi    75


# 4. 아래 레코드 추가하시오.
query = "INSERT INTO EXAM_TABLE VALUES('1005','2345','Jung',95)"
dbSendUpdate(conn,query)
query = "INSERT INTO EXAM_TABLE VALUES('1006','4567','Kang',80)"
dbSendUpdate(conn,query)


# 5. 추가된 레코드를 포함하여 exam_table 내 모든 데이터를 조회하시오.
query = "SELECT * FROM EXAM_TABLE"
dbGetQuery(conn, query)

# ID PASS NAME SCORE
# 1 1001 1234  Kim    90
# 2 1002 3456  Lee   100
# 3 1003 5678 Park    85
# 4 1004 7890 Choi    75
# 5 1005 2345 Jung    95
# 6 1006 4567 Kang    80


# 6. 성적(score) 기준으로 내림차순 정렬하시오.
query = "SELECT * FROM EXAM_TABLE ORDER BY SCORE DESC"
dbGetQuery(conn, query)

# ID PASS NAME SCORE
# 1 1002 3456  Lee   100
# 2 1005 2345 Jung    95
# 3 1001 1234  Kim    90
# 4 1003 5678 Park    85
# 5 1006 4567 Kang    80
# 6 1004 7890 Choi    75


# 7. name 이 ‘Choi’인 학생의 성적을 80점으로 수정하시오.
query = "UPDATE EXAM_TABLE SET SCORE = 80 WHERE NAME = 'Choi'"
dbSendUpdate(conn, query)

# ID PASS NAME SCORE
# 1 1001 1234  Kim    90
# 2 1002 3456  Lee   100
# 3 1003 5678 Park    85
# 4 1004 7890 Choi    80
# 5 1005 2345 Jung    95
# 6 1006 4567 Kang    80


# 8. 성적(score)이 80점 초과인 레코드만 조회하시오.
query = "SELECT * FROM EXAM_TABLE WHERE SCORE > 80"
dbGetQuery(conn, query)

# ID PASS NAME SCORE
# 1 1001 1234  Kim    90
# 2 1002 3456  Lee   100
# 3 1003 5678 Park    85
# 4 1005 2345 Jung    95


# 9. name이 ‘Kang’인 학생의 레코드를 삭제하시오.
query = "DELETE FROM EXAM_TABLE WHERE NAME = 'Kang'"
dbSendUpdate(conn, query)


# 10. 전체 레코드를 조회하시오.
query = "SELECT * FROM EXAM_TABLE"
dbGetQuery(conn, query)

# ID PASS NAME SCORE
# 1 1001 1234  Kim    90
# 2 1002 3456  Lee   100
# 3 1003 5678 Park    85
# 4 1004 7890 Choi    80
# 5 1005 2345 Jung    95