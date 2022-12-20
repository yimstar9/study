#임성구
##1번

data="a:b:c:d"
split=data.split(':')
split
'#'.join(split)

#2번
a = {'A':90, 'B':80}
# t=input("A or B를 입력하세요")
# try:
#
#     print(a[t])
# except KeyError:
#     print('70')
a.get('A',70)

#5번
#피보나치 수열
#정수n을 입력받았을때 n이하 까지 피보나치 수열 출력
def fibo(n):
    cnt=[0,1]
    i=0
    while True:
        cnt.append(cnt[i] + cnt[i + 1])
        if cnt[i+2] > n:
            del cnt[i+2]
            break
        i += 1
    print(cnt)
fibo(14)

#8번
import os
text=[]
print(os.getcwd())
fopen1=open(r'data\abc.txt',mode='r')
text=fopen1.read()
sorttxt=text.split('\n')
sorttxt.sort(reverse=True)
sorttxt='\n'.join(sorttxt)

fopen2=open(r'data\abc.txt',mode='w')
for i in sorttxt:
    fopen2.write(i)

fopen1.close()
fopen2.close()

#9번
import os
import math
import statistics
num=[]
print(os.getcwd())
fopen1=open(r'data\sample.txt',mode='r')
num=fopen1.read()
split = list(split)
split= list(map(int, num.split('\n')))
sub = sum(split)
avg = statistics.mean(split)
fopen2=open(r'data\result.txt',mode='w')
fopen2.write(f"총합 : {sub} 평균 : {avg}")
print(split)
print(sub,avg)

#10번  사칙연산 계산기
from statistics import mean
class Calculator():

    def __init__(self,a):
        self.num1=a
    def sum(self):
        print(sum(self.num1))
    def avg(self):
        print("%.1f" %mean(self.num1))

cal1 = Calculator([1,2,3,4,5])
cal1.sum()
cal1.avg()
cal2 = Calculator([6,7,8,9,10])
cal2.sum()
cal2.avg()

#12번
result = 0

try:
    [1, 2, 3, 4][3]
    "a"+1
    4 / 0
except TypeError:
    result += 1
except ZeroDivisionError:
    result += 2
except IndexError:
    result += 3
finally:
    result += 4

print(result)

#13번
data="4546793"
numlist=list(map(int,data))
result = []

for i, num in enumerate(numlist):
    result.append(str(num))
    if i <len(numlist)-1:
        odd=num%2==1
        odd1=numlist[i+1]%2==1
        if odd and odd1:
            result.append('-')
        elif not odd and not odd1:
            result.append('*')
print("".join(result))

#14번
def compress_string(s):
    _c = ""
    cnt = 0
    result = ""
    for c in s:
        if c!=_c:
            _c = c
            if cnt: result += str(cnt)
            result += c
            cnt = 1
        else:
            cnt +=1
    if cnt: result += str(cnt)
    return result

print(compress_string("aaabbcccccca"))