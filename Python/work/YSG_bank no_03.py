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