################################
#54321 출력
def Counter(n):
    if n==0:
        return 0
    else:
        print(n)
        Counter(n-1)
Counter(5)

#############################
#12345 출력  두개가 왜 순서가 차이나는지 모르겠다
def Counter(n):
    if n==0:
        return 0
    else:
        Counter(n-1)
        print(n)
Counter(5)
##############################
#1~n 누적합 (꼬리 재귀 안씀)
def Adder(n):
    if n==1:
        return 1
    else:
        result =n+Adder(n-1)
        print(n, end=' ')
        return result
print(Adder(10))
print(Adder(2))
###############################
#1~n누적합
def Adder(n):
    if n==1:
        return 1
    else:
        # print(n, end=' ')
        return n+Adder(n-1) ###이러면 메모리 사용량 많아져서 매우 느려진다.
print(Adder(10))
print(Adder(2))
#################################
#1~n 누적합(꼬리재귀)
def Adder(n,total=0):
    if n==0:
        return total
    else:
        return Adder(n-1,n+total) #리턴값에 연산을 넣지 않아서 재귀함수의 단점(느려짐)을 보완
        # print(n, total, end=' ')
print(Adder(3))
