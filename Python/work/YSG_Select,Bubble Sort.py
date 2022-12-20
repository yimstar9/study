#임성구
#선택정렬 오름차순
dataset=[3,5,1,2,4]
l=len(dataset)

for i in range(l-1):
    minIndex=i
    for j in range(i+1,l):
      if dataset[j]<dataset[minIndex]:
        minIndex = j
    print(dataset)
    dataset[i],dataset[minIndex] =dataset[minIndex],dataset[i]
print(dataset)

#선택정렬 내림차순
dataset=[3,5,1,2,4]
l=len(dataset)
for i in range(l-1):
    maxIndex=i
    for j in range(i+1,l):
      if dataset[j]>dataset[maxIndex]:
        maxIndex = j
    dataset[i],dataset[maxIndex] =dataset[maxIndex],dataset[i]
    print(dataset)
print(dataset)

#버블정렬 오름차순
dataset=[5,2,4,3,1]
l=len(dataset)
for i in range(l-1):
    for j in range(l-i-1):
      if dataset[j]>dataset[j+1]:
        print(dataset)
        dataset[j],dataset[j+1] =dataset[j+1],dataset[j]
print(dataset)

#버블정렬 내림차순
dataset=[3,5,7,8,11,42]
l=len(dataset)
for i in range(l-1):
    for j in range(l-i-1):
      if dataset[j]<dataset[j+1]:
        print(dataset)
        dataset[j],dataset[j+1] =dataset[j+1],dataset[j]
print(dataset)

