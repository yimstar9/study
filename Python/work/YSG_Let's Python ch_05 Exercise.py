#임성구
#1번
def StarCount (height):
    cnt=total=0
    while cnt < height:
        cnt += 1
        print('*' * cnt)
        total+=cnt
    return total
height = int(input('height : '))
print('star개수: %d'%StarCount(height))

#2번
def bank_account(bal):
    balance=bal
    def getBalance():
        return balance
    def deposit(money):
        nonlocal balance
        balance += money
        print(f'{money}원 입금후 잔액은 {getBalance()}원 입니다.')
    def withdraw(money):
        nonlocal balance
        if balance>=money:
            balance-= money
            print(f'{money}원 출금후 잔액은 {getBalance()}원 입니다.')
        else:
            print('잔액이 부족합니다.')
    return getBalance,deposit,withdraw

getBalance,deposit,withdraw = bank_account(int(input("최초 계좌의 잔액을 입력하세요 : ")))
print(f'현재 계좌 잔액은 {getBalance()}원 입니다.')
deposit(int(input("입금액을 입력하세요 : ")))
withdraw(int(input("출금액을 입력하세요 : ")))

#3번

def Factorial(n,total=1):
    if n==1:
        return total
    else:
        return Factorial(n-1,n*total)
result_fact =Factorial(int(input("숫자를 입력하세요:")))
print('패토리얼 결과:', result_fact)