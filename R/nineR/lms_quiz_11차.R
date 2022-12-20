##2
Sys.setenv(JAVA_HOME = "C:\\Program Files\\Java\\jdk1.8.0_202")

library(DBI)
library(rJava)
library(RJDBC)

remove.packages("rJava", lib="~/R/win-library/4.0")
drv <- JDBC("oracle.jdbc.driver.OracleDriver", "C:/OracleTest/ojdbc6.jar")
conn <- dbConnect(drv,"jdbc:oracle:thin:@//127.0.0.1:1521/xe", "scott2", "tiger")


##3
query = "SELECT * FROM EXAM_TABLE"
dbGetQuery(conn, query)

##4
query = "INSERT INTO EXAM_TABLE VALUES('1005','2345','Jung',95)"
dbSendUpdate(conn, query)
query = "INSERT INTO EXAM_TABLE VALUES('1006','4567','Kang',80)"
dbSendUpdate(conn, query)

##5
query = "SELECT * FROM EXAM_TABLE"
dbGetQuery(conn, query)

##6
query = "SELECT * FROM EXAM_TABLE ORDER BY SCORE DESC"
dbGetQuery(conn, query)

##7
query = "UPDATE EXAM_TABLE SET SCORE = 80 WHERE NAME = 'Choi'"
dbSendUpdate(conn, query)

##8
query = "SELECT * FROM EXAM_TABLE WHERE SCORE > 80"
ans <- dbGetQuery(conn, query)
ans

##9
query = "DELETE FROM EXAM_TABLE WHERE NAME = 'Kang'"
dbSendUpdate(conn, query)

##10
query = "SELECT * FROM EXAM_TABLE"
dbGetQuery(conn, query)