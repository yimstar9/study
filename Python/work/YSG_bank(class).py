class Account:
    __balance = 0

    def __init__(self, balance, deposit, withdraw):
        self.__balance = balance
        self.__deposit = deposit
        self.__withdraw = withdraw
        print(f'현잔고 : {self.__balance:,d}원')

    def calc(self):
        self.__balance += self.__deposit
        print(f'입금후 잔고 : {self.__balance:,d}원')
        if self.__balance < self.__withdraw:
            print('잔액부족')
            return
        self.__balance -= self.__withdraw
        print(f'출금후 잔고 : {self.__balance:,d}원')

acc = Account(10000,5000,8000)
acc.calc()


####
#2번
from math import sqrt
class Account:
    __balance = 0

    def __init__(self, balance, deposit, withdraw):
        self.__balance = balance
        self.__deposit = deposit
        self.__withdraw = withdraw
        print(f'현잔고 : {self.__balance:,d}원')

    def calc(self):
        self.__balance += self.__deposit
        print(f'입금후 잔고 : {self.__balance:,d}원')
        if self.__balance < self.__withdraw:
            print('잔액부족')
            return
        self.__balance -= self.__withdraw
        print(f'출금후 잔고 : {self.__balance:,d}원')
    def sqrt1(self):
        self.sqrresult = sqrt(self.__balance)
        return self.sqrresult

acc = Account(10000, 5000, 8000)
acc.calc()
val=acc.sqrt1()

print('sqrt값 :',val)


#3번
from math import sqrt
class Account:
    __balance = 0

    def __init__(self, balance, deposit, withdraw):
        self.__balance = balance
        self.__deposit = deposit
        self.__withdraw = withdraw
        print(f'현잔고 : {self.__balance:,d}원')

    def calc(self):
        self.__balance += self.__deposit
        print(f'입금후 잔고 : {self.__balance:,d}원')
        if self.__balance < self.__withdraw:
            print('잔액부족')
            return
        self.__balance -= self.__withdraw
        print(f'출금후 잔고 : {self.__balance:,d}원')
    def sqrt1(self):
        self.sqrtresult = sqrt(self.__balance)
        return self.sqrtresult
    def mul(self,num):
        result1 = self.sqrtresult*num
        return result1
acc = Account(10000, 5000, 8000)
acc.calc()
val=acc.sqrt1()
mul=acc.mul(3)
print('sqrt값 :',val)
print('sqrt에 곱한값 :',mul)