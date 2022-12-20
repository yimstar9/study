#01-A형 항공사에서는 짐을 부칠때 10kg이상이면 수수료 만원, 만약 10kg미만이면 수수료x
#      사용자의 짐의 무게를 키보드로 입력 받아서 사용자가 지불하여야 할 금액을 계산하는 프로그램
Weight=int(input("짐의 무게는 얼마입니까?"))
if Weight < 10:
    print('수수료는 없습니다.')
else:
    print('수수료는 10,000원 입니다.')

#01-B형 수수료는 10의 배수 단위로 만원씩 증가한다 10kg미만은 수수료없다.
Weight=int(input("짐의 무게는 얼마입니까?"))
over=Weight//10
if Weight < 10:
    print('수수료는 없습니다.')
else:
    print(f'수수료는 {over*10000:,d}원 입니다.')

#02 1~10 사이 난수 맞추기 게임
import random
print('>>1~10 숫자 맞추기 게임<<')
com = random.randint(1,10)
while True:
     my = int(input("예상 숫자(1~10)를 입력하시오:"))
     if my == com:
         print("~~성공~~")
         break
     elif my > com:
         print("더 작은수 입력")
     else:
         print("더 큰수 입력")

#03 1~100 사이에 3의 배수 and 2의 배수 아닌 수 합, 숫자리스트
cnt=tot=0
list=[]
for cnt in range(100):
    if cnt%3 ==0 and cnt%2 !=0:
        tot+= cnt
        list.append(cnt)
print(list)
print(f"누적합 : {tot}")

#04 단어의 개수를 출력하시오
multilin="""안녕하세요. 파이썬 세계로 오신걸 
환영합니다.
파이썬은 비단뱀 처럼 매력적인 언어입니다."""
words=[]
for word in multilin.split():
    words.append(word)
# print(f"단어: {words}")
for i in words:
    print(i)
print("단어수:", len(words))
