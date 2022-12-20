#임성구
#1번
def bank_account(num1,d,w):
    balance=num1
    dep=d
    withd=w
    def calc():
        nonlocal balance
        balance += dep
        balance-= withd
        return balance
    return calc

calc = bank_account(10000,5000,8000)
print("잔고는 %d원 입니다."%calc())

#2번
from math import sqrt
def bank_account(n,d,w):
    balance=n
    dep=d
    withd=w
    sq =0
    def out():
        return balance,sq
    def calc():
        nonlocal balance
        nonlocal sq
        balance += dep
        balance-= withd
        sq = sqrt(balance)
    return out,calc

out,calc = bank_account(10000,5000,8000)
calc()
print("잔고는 %d원 sqrt값은 %f 입니다."%(out()[0],out()[1]))

#3번
from math import sqrt
def bank_account(n,d,w,m):
    balance=n
    dep=d
    withd=w
    sq =0
    mul=m
    def out():
        return balance,sq
    def calc():
        nonlocal balance
        nonlocal sq
        balance += dep
        balance-= withd
        sq = sqrt(balance)
    def multi():
        nonlocal sq
        mult = sq*mul
        return mult
    return out,calc,multi

out,calc,multi = bank_account(10000,5000,8000,3)
calc()
print("잔고는 %d원 sqrt값은 %f 곱한값은 %f입니다."%(out()[0],out()[1],multi()))

#3번-1
from math import sqrt
def bank_account(n,d,w,m):
    balance=n
    dep=d
    withd=w
    mul=m
    sq=0
    def sqrt1(val):
        nonlocal sq
        sq = sqrt(val)
        return sq
    def bal():
        nonlocal balance
        balance += dep
        balance-= withd
        return balance
    def multi():
        nonlocal sq
        mult = sq*mul
        return mult
    return bal,sqrt1,multi

bal1,sqrt1,multi = bank_account(10000,5000,8000,3)
b=bal1()
print("잔고는 %d원 sqrt값은 %f 곱한값은 %f입니다."%(b,sqrt1(b),multi()))