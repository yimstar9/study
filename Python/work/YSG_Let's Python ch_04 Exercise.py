#임성구
#1번 다음 1st변수를 대상으로 각 단계별로 list를 연산하시오
#1st=[10,1,5,2]
#단계1 : 1st원소를 2배 생성하여 result변수에 저장 및 출력
#단계2 : 1st첫번째 원소에 2를 곱하여 result 변수에 추가 및 출력
#단계3 : result의 홀수 번째 원소만 result2변수에 추가 및 출력
lst = [10,1,5,2]
i=0
result = []
result2 = []
result = lst*2
print('단계1:',result)
result.append(lst[0]*2)
print('단계2:',result)
result2=[result[i] for i in range(len(result)) if i%2==1]
print('단계3:',result2)

#2번
import random
r=[]
a=int(input("vector 수:"))
for i in range(a):
    num =random.random()*100
    r.append(f'{num:.0f}')
    print(f'{num:.0f}')
# print(r)
print("vector크기:",len(r))

#2번-B형
import random
r=[]
a=int(input("vector 수:"))
for i in range(a):
    num = random.random() * 100
    r.append(round(num)) #round함수 소수점 반올림
    print(round(num))
# print(r)
b=int(input("vector 내 찾는값:"))
if b in r:
    print("YES")
else:
    print("NO")

#3번-A형
message = ['spam','ham','spam','ham','spam']
dummy=[1 if p=="spam" else 0 for p in message]
print(dummy)
#3번-B형
message = ['spam','ham','spam','ham','spam']
spam_list=[key for key in message if key=='spam']
print(spam_list)

#4번
position = ['과장', '부장','대리','사장','대리','과장']
positionset=list(set(position))
cnt ={}
for key in position:
    cnt[key]=cnt.get(key,0)+1
print('중복되지 않은 직위:',positionset)
print('각 직위별 빈도수:',cnt)

