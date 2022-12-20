#임성구
cnt=0
tot=0
while cnt < 100:
    cnt+=1
    if cnt %5 ==0 and cnt%3 !=0:
        tot += cnt
print(f'1~100 사이 5의배수 이면서 3의 배수 아닌 수 합 : {tot}')
