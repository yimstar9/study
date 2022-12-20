help (map)
set.add

round(2.5)
help(round)

def bank_account(bal):
    balance=bal
    def getBalance():
        return balance
    def deposit(money):
        nonlocal balance
        balance += money
        print(f'{money}원 입금후 잔액은 {getBalance()}원 입니다.')
    def withdraw(money):
        nonlocal balance
        if balance>=money:
            balance-= money
            print(f'{money}원 출금후 잔액은 {getBalance()}원 입니다.')
        else:
            print('잔액이 부족합니다.')
    return getBalance,deposit,withdraw

getBalance,deposit,withdraw = bank_account(int(input("최초 계좌의 잔액을 입력하세요 : ")))
print(f'현재 계좌 잔액은 {getBalance()}원 입니다.')
deposit(int(input("입금액을 입력하세요 : ")))
withdraw(int(input("출금액을 입력하세요 : ")))

def pytha(s,t):
    a= s**2-t**2
    b=2*s*t
    c=s**2+t**2
    print("3변의 길이:",a,b,c)

pytha(6,1)

import random
def coin(n):
    result = []
    for i in range(n):
        r=random.randint(0,1)
        if(r==1): result.append(1)
        else: result.append(0)
        return result
    print(coin(10))
def montaCoin(n):
    cnt = 0
    for i in range(n):
        cnt+=coin(1)[0]
    result = cnt/n
    return result
print(montaCoin(100))


####
data = list(range(1,101))
def outer_func(data):
    dataset= data
    def tot():
        tot_val = sum(dataset)
        return tot_val
    def avg(tot_val):
        avg_val = tot_val/len(dataset)
        return avg_val
    return tot,avg
tot,avg = outer_func(data)

tot_val = tot()
print("tot = ",tot_val)
avg_val= avg(tot_val)
print("avg = ",avg_val)

####
from statistics import  mean
from math import sqrt

data = [4,5,3.5, 2.5, 6.3, 5.5]

def scattering_func(data):
    dataSet = data

    def avg_func():
        avg_val = mean(dataSet)
        return avg_val
    def var_func(avg):
        diff=[(data-avg)**2 for data in dataSet]
        #print(sum(diff))
        var_val = sum(diff)/(len(dataSet)-1)
        return var_val
    def std_func(var):
        std_val = sqrt(var)
        return std_val

    return avg_func, var_func, std_func

avg,var,std=scattering_func(data)

print('평균:',avg())
print('분산:',var(avg()))
print('표준편차:',std(var(avg())))

##
def main_func(num):
    num_val = num
    def getter():
        return num_val
    def setter(value):
        nonlocal num_val
        num_val = value
       # print(num_val)
    return getter, setter

getter, setter = main_func(100)
print('num=',getter())
setter(200)
print('num=',getter())


def wrap(func):
    def decorated():
        print('반갑습니다.!')
        func()
        print('잘가요')
    return decorated

@wrap
def hello():
    print('hi~', '홍길동')

hello()

def Counter(n):
    if n ==0:
        return 0
    else :

        Counter(n-1)
        print(n)
print('n=0:', Counter(0))
Counter(5)


