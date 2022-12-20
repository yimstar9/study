
x=[2,4,1,5,7]

lst=[i**2 for i in x] # x원소에 제곱 계산
print(lst)


x=[2,4,1,5,7]

num=list(range(1,11))

lst2=[i*2 for i in num if i %2 ==0]

print(lst2)


charset = ['가','나','다','가','다','라','가']

wc={} #빈 셋 선언


for key in charset:

    wc[key] = wc.get(key,0) +1 #get()함수를 이용하여 키에 해당하는 값을 꺼내온다. 이때 값이 없는경우 (최초로 발견된 단어)0으로 초기화 하고 1을 더해서 값을 만든다.

print(wc)




charset = ['가','나','다','가','다','라','가']
wc={} #빈 셋 선언

for key in charset:
    if key in wc:
        wc[key] += 1
    else:
     wc[key]=1
print(wc)

from copy import copy
name=['홍','이','감']
name2=copy(name)
print(name is name2)


#이진 검색 알고리즘
dataset=[5,10,18,22,35,55,75,103]
value = int(input("검색할 값 입력:"))

low =0 #start위치
high = len(dataset)-1 #end 위치
loc = 0
state = False #상태변수

while( low <= high):
    mid = (low + high) //2

    if dataset[mid]>value: #중앙값이 큰 경우
        high = mid -1
    elif dataset[mid]<value:
        low = mid +1
    else:
        loc = mid
        state =True
        break

if state:
    print("찾은위치 : %d 번째"%(loc+1))
else:
    print('찾는값은 없습니다.')
