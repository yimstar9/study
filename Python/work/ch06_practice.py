class Dog:  # 클래스 선언

    name = "삼식이"

    age = 3

    breed = "골든 리트리버"

    def bark(self):
        print(self.name + "가 멍멍하고 짖는다.")


my_dog = Dog()  # 인스턴스 생성

print(my_dog.breed)  # 인스턴스의 속성 접근

my_dog.bark()  # 인스턴스의 메소드 호출

################################################
#함수와 클래스
def cal_func(a,b):
    x=a
    y=b
    def plus():
        p=x+y
        return p
    def minus():
        m=x-y
        return m
    return plus, minus

p1,m1 = cal_func(10,20)
print('더하기:',p1())
print('빼기:',m1())

#클래스
class cal_class:
    x=y=0
    #생성자 : 객체 생성+멤버 변수 초기화
    def __init__(self,a,b):
        self.x=a
        self.y=b
    def plus(self):
        p=self.x+self.y
        return p
    def minus(self):
        m=self.x-self.y
        return m

obj = cal_class(10,20)
print('더하기:',obj.plus())
print('빼기:',obj.minus())


#################################
#상속
class Super:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def display(self):
        print('name:%s,age:%d'%(self.name,self.age))

sup= Super('부모',55)
sup.display()

class Sub(Super):
    gender=None
    def __init__(self,name,age,gender):
        self.name = name
        self.age = age
        self.gender=gender
    def display(self):
        print('name:%s,age:%d,gender:%s' % (self.name, self.age,self.gender))

sub =Sub('자식',25,'여자')
sub.display()

#########################
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val

class UpgradeCalculator(Calculator):

    def minus(self,val):
        self.value -=val
cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)
print(cal.value)



#######################
class Account:
    __balance = 0
    __accName=None
    __accNo= None
    
    def __init__(self,bal,name,no):
        self.__balance = bal 
        self.__accName = name #예금주
        self.__accNo = no #계좌번호
    def getBalace(self):
        return self.__balance,self.__accName,self.__accNo
    
    def deposit(self, money):
        if money<0:
            print('금액확인')
            return #종료
        self.__balance+=money
    def withdraw(self, money):
        if self.__balance<money:
            print('잔액부족')
            return #종료
        self.__balance-=money
    
acc=Account(1000,'홍길동', '12-152-4125-41')

bal = acc.getBalace()
print('계좌정보:',bal)

acc.deposit(10000)
bal = acc.getBalace()
print('계좌정보:',bal)

acc.withdraw(10000)
bal = acc.getBalace()
print('계좌정보:',bal)

###########################
class Parent:
    def __init__(self,name,age):
        self.name = name
        self.age= age
        
    def dispaly(self):
        print('name: %s, age:%d' %(self.name, self.age))
sup = Parent('부모',55)
sup.dispaly()

class Sub(Parent):
    gender = None
    def __init__(self,name,age,gender):
        #self.name = name
        #self.age = age
        super().__init__(name,age)
        self.gender= gender
    def display(self):
        print('name : %s, age = %d, gender %s' %(self.name, self.age, self.gender))
sub = Sub('자식', 25, '여자')
sub.display()


##########################
class Flight:
    def fly(self):
        print('날다,fly 원형 메서드')

class Airplane(Flight):
    def fly(self):
        print('비행기가 날다')
class Bird(Flight):
    def fly(self):
        print('새가 날다')
class paperAirplane(Flight):
    def fly(self):
        print('종이 비행기가 날다')

flight = Flight()
air = Airplane()
bird = Bird()
paper = paperAirplane()

flight.fly()
flight = air
flight.fly()