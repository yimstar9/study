#임성구
#-1,3,-5,7,-9,~-97,99의 합을 구하시오
cnt=0
tmp=0
tot=0
dataset=[]
while cnt < 50:
    cnt+=1
    tmp = (-1)**cnt*(cnt * 2 - 1)
    tot+=tmp
    dataset.append(tmp)
print(dataset)
print(tot)