#임성구

#1번-readline함수
# word=[]
# ftest = open(r'ch8_data\ch8_data\data\ftest.txt',mode='r')
# docs=ftest.readlines()
# docs=[line.strip('\n') for line in docs]
# print('문장내용\n',docs)
# print('문장수 :',len(docs))
# for i in docs:
#     word.extend(i.split(' '))
# print('단어내용\n',word)
# print('단어수 :',len(word))

#1번-read함수
import os
os.getcwd()
word=[]
ftest = open(r'ch8_data\ch8_data\data\ftest.txt',mode='r')
docs=ftest.read().split(('\n'))
print('문장내용\n',docs)
print('문장수 :',len(docs))
for i in docs:
    word.extend(i.split(' '))
print('단어내용\n',word)
print('단어수 :',len(word))


#2번
import pandas
from statistics import mean
import os
os.getcwd()
emp = pandas.read_csv("ch8_data/ch8_data/data/emp.csv", encoding='utf-8')
no = emp.No
name = emp.Name
pay = emp.Pay
print("관측치 길이 :",len(emp))
print("전체 평균 급여 : %.1f"%(mean(pay)))

for j,p in enumerate(pay):
    if p == min(pay):
        print(f"최저 급여 : {p} 이름 : {name[j]}")
    elif p == max(pay):
        print(f"최고 급여 : {p} 이름 : {name[j]}")


