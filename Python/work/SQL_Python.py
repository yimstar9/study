import cx_Oracle
import os
import pandas as pd
import sqlite3

connection = sqlite3.connect('./stocks.sqlite')

os.putenv('NLS_LANG', '.UTF8')

LOCATION = r"C:/instantclient_21_7"
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]

connect = cx_Oracle.connect("scott","tiger", "localhost:1521/xe")

cs = connect.cursor()
# 데이터 추가(insert)
sql = "insert into py_table values('kang', '1234', '강감찬', 45)"
cs.execute(sql) # SQL 문
cs.execute("select * from py_table") # SQL 문
for i in cs: # data 보기
 print(i)
# 나이가 40 세 이상인 record
sql = "select * from test_table where age >= 40"
cs.execute(sql)
for i in cs: # data 보기
 print(i)
# name 이 '강감찬'인 데이터의 age 를 40 으로 수정
sql = "update test_table set age = 40 where name = '강감찬'"
cs.execute(sql)
sql = "select * from test_table where name = '강감찬'"
cs.execute(sql)
# name 이 '홍길동'인 레코드 삭제
sql = "delete from test_table where name = '홍길동'"
cs.execute(sql)
# 전체 레코드 조회
sql = "select * from test_table"
cs.execute(sql)
for i in cs: # data 보기
 print(i)

################################################
# Python Pandas : pandas.io.sql.get_schema
# (DataFrame 내용을 sql create table syntax로 만들기)
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql_query.html
################################################

dict_test = {
    'col1': [1, 2, 3, 4, 5],
    'col2': ['a', 'b', 'c', 'd', 'e'],
    'col3': [6, 7, 8, 9, 10],
    'col4': [1, 2, 3, 'b', 'c']
}

df_test = pd.DataFrame(dict_test)
print(df_test)
sql_df=pd.io.sql.get_schema(df_test, name='df_test')
print(sql_df)
val_schema = sql_df.replace('\"', '')
val_schema = sql_df.replace('col1', 'col1')
val_schema = sql_df.replace('col2', 'col2')
val_schema = sql_df.replace('col3', 'col3')
val_schema = sql_df.replace('col4', 'col4')
val_schema = sql_df.replace("TEXT", 'varchar')
val_schema = sql_df.replace('INTEGER', 'int')
print(val_schema)
cs.execute(val_schema)

##########https://greendreamtrre.tistory.com/223
from sqlalchemy import create_engine
#sqlalchemy 패키지 설치
engine = create_engine('oracle+cx_oracle://scott2:tiger@localhost:1521/xe')
df = pd.read_csv('2021년 누적 뺀거.csv',encoding='cp949')
df.columns = ['country','c1','d1','c2','d2']
df.to_sql(name="con",con=engine, if_exists='replace',index=True)
readpd = pd.read_sql_query('SELECT * FROM con', engine)
readpd

cs.close()
connect.commit()
connect.close()