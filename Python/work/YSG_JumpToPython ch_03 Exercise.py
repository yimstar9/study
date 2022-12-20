#1번 다음 코드값의 결과는?
a = "Life is too short, you need python"

if "wife" in a: print("wife")   # wife가 포함되어 있으면 wife프린트
elif "python" in a and "you" not in a: print("python") #python이 포함되고 you가 포함되어 있으면 python프린트
elif "shirt" not in a: print("shirt") #shirt가 포함되어 있지 않으면 shirt프린트
elif "need" in a: print("need") #need가 포함되어 있으면 need프린트
else: print("none")
#shirt 출력이 된다. need도 만족하지만 shirt줄이 먼저 만족해서 shirt출력하고 빠져나왔다.

#2번 while문을 사용해 1부터 1000까지의 자연수 중 3의 배수의 합을 구해 보자.
cnt=sum=0

while True:
    cnt+=1
    if cnt%3==0:
        sum+=cnt
    elif cnt>1000:
        break
print(sum)

#3번 while문을 사용하여 다음과 같이 별(*)을 표시하는 프로그램을 작성해 보자.
i = 0
while True:
    i += 1
    if (i > 5): break
    print('*'*i)


#4번 for문을 사용해 1부터 100까지의 숫자를 출력해 보자.
for i in range(1,101):
    print(i)

#5번 A 학급에 총 10명의 학생이 있다. 이 학생들의 중간고사 점수는 다음과 같다.
# [70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
# for문을 사용하여 A 학급의 평균 점수를 구해 보자.
med=sum=0
list=[70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
for i in list:
    sum+=i
med=sum/len(list)
print(med)

#6번 리스트 중에서 홀수에만 2를 곱하여 저장하는 다음 코드가 있다.
# numbers = [1, 2, 3, 4, 5]
# result = []
# for n in numbers:
#     if n % 2 == 1:
#         result.append(n*2)
# 위 코드를 리스트 내포(list comprehension)를 사용하여 표현해 보자.
numbers=[1,2,3,4,5]
result=[i*2 for i in numbers if i%2==1]
print(result)