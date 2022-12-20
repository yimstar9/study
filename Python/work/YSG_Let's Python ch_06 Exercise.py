#임성구
#1번
class Rectangle:
    width=height=0
    def __init__(self,width,height):
        self.width=width
        self.height=height
    def area_calc(self):
        area = self.width*self.height
        return area
    def circum_calc(self):
        circum = 2*(self.width+self.height)
        return circum


print("사각형의 넓이와 둘레를 계산합니다")
w=int(input("사각형의 가로 입력:"))
h=int(input("사각형의 세로 입력:"))
print('-'*30)
rect=Rectangle(w,h)
print("사각형의 넓이:", rect.area_calc())
print("사각형의 둘레:", rect.circum_calc())
print('-'*30)
#2번
from statistics import mean
from math import sqrt
x=[5,9,1,7,4,6]
class Scattering:
    def __init__(self,x):
        self.lst=x
    def var_func(self):
        lst=[(i-mean(self.lst))**2 for i in self.lst]
        self.v=sum(lst)/(len(self.lst)-1)
        return self.v
    def std_func(self):
        std=sqrt(self.v)
        return std

cal = Scattering(x)
print("분산:",cal.var_func())
print("표준편차:",cal.std_func())

#3번
class person:
    name=age=0
    gender=None
    def __init__(self,name,gender,age):
        self.name=name
        if gender=="male":
            self.gender="남자"
        if gender=="female":
            self.gender = "여자"
        else:
            self.gender="알 수 없음"
        self.age=age

    def display(self):
        print("="*20)
        print(f'이름:{self.name}, 성별:{self.gender}')
        print("나이:%d"%self.age)
        print("=" * 20)

name=input("이름입력:")
age=int(input("나이 입력"))
gender=input("성별(male/female)입력:")
p=person(name,gender,age)
p.display()

#4번
class Employee:
    name = None
    def __init__(self,name):
        self.name=name

class Permanet(Employee):
    def __init__(self,name):
        super().__init__(name)

    def pay_calc(self,normal,bonus):
        self.pay = normal+bonus
        print('=' * 30)
        print("고용형태 : 정규직")
        print("이름 : ",self.name)
        print("급여 : ",format(self.pay,',d'),'원')

class Temporary(Employee):
    def __init__(self,name):
        super().__init__(name)

    def pay_calc(self,time1,hour):
        self.pay = time1*hour
        print('='*30)
        print("고용형태 : 임시직")
        print("이름 : ",self.name)
        print("급여 : ", format(self.pay, ',d'), '원')

empType = input("고용형태 선택(정규직<P>, 임시직<T>) : ")
if empType == 'P' or empType == 'p' :
    name=input("이름 : ")
    normal=int(input("기본급 : "))
    bonus=int(input("상여금 : "))
    p = Permanet(name)
    p.pay_calc(normal,bonus)
elif empType == 'T' or empType == 't':
    name=input("이름 : ")
    time1=int(input("작업시간 : "))
    hour=int(input("시급 : "))
    e = Temporary(name)
    e.pay_calc(time1,hour)
else:
    print('='*30)
    print('입력오류')


#5번
from myCalcPackage.calcModule import Add,Sub,Mul,Div
x=10
y=5
print("Add=",Add(x,y))
print("Sub=",Sub(x,y))
print("Mul=",Mul(x,y))
print("Div=",Div(x,y))