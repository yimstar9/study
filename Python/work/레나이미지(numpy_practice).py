import numpy as np
import pandas as pd

arr1 = []
arr1=np.zeros((2,2,3,3))
arr1
arr1.ndim


##
import numpy as np
import random
arr1=[]
r=random.seed()
r

##image matrix

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
image = Image.open("data/Lenna.png")
img_color=np.array(image)
timg = np.zeros(np.shape(img_color))
z = np.zeros([32,32,3])
img_gray = np.zeros(np.shape(img_color))
count=0
maxV=0

#for문 써서 구현
start = datetime.now()

for i in range(len(img_color)):
    for j in range(len(img_color[0])):
        img_gray[i,j]=sum(img_color[i,j])/3
        count += 1

img_gray.astype(np.uint8)
img_gray= img_gray.astype(np.uint8)
# z= z.astype(np.uint8)
# timg=timg.astype(np.uint8)
plt.subplot(2,2,1),plt.imshow(img_color)
plt.subplot(2,2,2),plt.imshow(img_gray)
# plt.subplot(2,2,4),plt.imshow(timg)
# plt.subplot(2,2,3),plt.imshow(z)
plt.show()
print(datetime.now() - start)

image.close()


#
# for i in range(len(img_color)):
#     for j in range(len(img_color[0])):
#
#         img_gray[i,j]=sum(img_color[i,j])/3
#         # timg[i,j] = np.where(img_gray[i,j]>123, 255, 0)
#         # if count%8 ==0:
#         #     a=i//16
#         #     b=j//16
#         #     z[a,b]=img_color[i,j]
#         count += 1

###repeat 함수 써서 for문 대신 해보기###########
###동작 속도 훨씬 빨라진다.
start = datetime.now()
nshape1=img_color
means = np.mean(nshape1, axis = 2)
final_result = np.repeat(means, 3, axis=1)
img_gray = final_result.reshape(512,512,3)


img_gray.astype(np.uint8)
img_gray= img_gray.astype(np.uint8)
# z= z.astype(np.uint8)
# timg=timg.astype(np.uint8)
plt.subplot(2,2,1),plt.imshow(img_color)
plt.subplot(2,2,2),plt.imshow(img_gray)
# plt.subplot(2,2,4),plt.imshow(timg)
# plt.subplot(2,2,3),plt.imshow(z)
plt.show()
print(datetime.now() - start)

image.close()


###tile함수 써서 for문 대신 해보기###########

start = datetime.now()
nshape1=img_color
means = np.mean(nshape1, axis = 2)

b1=np.tile(means.reshape(-1,1),(1,3))
img_gray = b1.reshape(512,512,3)

img_gray.shape
img_gray.astype(np.uint8)
img_gray= img_gray.astype(np.uint8)
# z= z.astype(np.uint8)
# timg=timg.astype(np.uint8)
plt.subplot(2,2,1),plt.imshow(img_color)
plt.subplot(2,2,2),plt.imshow(img_gray)
# plt.subplot(2,2,4),plt.imshow(timg)
# plt.subplot(2,2,3),plt.imshow(z)
plt.show()
print(datetime.now() - start)

image.close()

#######broadcast_to함수

start = datetime.now()
nshape1=img_color
means = np.mean(nshape1, axis = 2)
b1=np.broadcast_to(means, (3,512,512))
b1=np.transpose(b1)
img_gray = b1


img_gray.astype(np.uint8)
img_gray= img_gray.astype(np.uint8)
# z= z.astype(np.uint8)
# timg=timg.astype(np.uint8)
plt.subplot(2,2,1),plt.imshow(img_color)
plt.subplot(2,2,2),plt.imshow(img_gray)
# plt.subplot(2,2,4),plt.imshow(timg)
# plt.subplot(2,2,3),plt.imshow(z)
plt.show()
print(datetime.now() - start)

image.close()

means[:,:,None].shape

############################
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]
# arr3d[0]에는 스칼라값과 배열 모두 대입 가능
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d
# arr3d[1, 0]은 (1,0)으로 색인되는 1 차원 배열과 그 값 반환
arr3d[1, 0]
# 이 값은 아래의 결과와 동일
x = arr3d[1]
x
x[0]

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4) # 7x4 array
names
data
names == 'Bob'
data[names == 'Bob']


# Fancy Indexing 은 정수 배열을 사용한 색인을 설명하기 위함.
# 8x4 배열
arr = np.empty((8, 4)) # 8x4 array
for i in range(8):
 arr[i] = i
arr
arr[2]=3

# 다차원 색인 배열을 넘기는 것은 다르게 동작. 각각의 색인 튜플에 대응하는 1 차원 배열이 선택됨.
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]] # 인덱스 (1,0) (5,3) (7,1) (2,2)
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

arr = np.ones([3,5])
arr
arr.T
np.dot(arr.T, arr)
arr.T@arr


####
# np.linalg: 행렬의 분할과 역행렬, 행렬식과 같은 것 포함.
from numpy.linalg import inv, qr
# X = np.random.randn(2, 2)
X= np.array([[2,4],[4,2]])
X2=[6,6]
inv(X).dot(X2)
X.T
mat = X.T.dot(X)
mat
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
q
r


# 순수 파이썬 내 내장 random 모듈 사용하여 계단 오르내리기를 1,000 번 수행하는 코드:
import random
import matplotlib.pyplot as plt
position = 0
walk = [position]
steps = 1000
for i in range(steps):
 step = 1 if random.randint(0, 1) else -1
 position += step
 walk.append(position)
plt.figure()
# 처음 100 회 계단 오르내리기를 그래프화
plt.plot(walk[:1000])
# 1,000 번 수행
np.random.seed(12345)
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
# 계단을 오르내린 위치의 최소값과 최대값
walk.min()
walk.max()
# 최초의 10 혹은 -10 인 시점
(np.abs(walk) >= 10).argmax()
# 4.7.1 Simulating Many Random Walks at Once
# np.random 함수에 크기가 2 인 튜플을 넘기면 2 차원 배열이 생성
# 각 컬럼에서 누적합을 구해서 5,000 회의 시뮬레이션을 한번에 처리
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks
26
# 모든 시뮬레이션에 대해 최대값과 최소값
walks.max()
walks.min()
# 누적합이 30 또는 -30 에 도달하는 최소시점 계산
hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum() # Number that hit 30 or -30
# 처음 위치에서 30 칸 이상 멀어지는 최소 횟수:
# 컬럼 선택하고 절대값이 30 을 넘는 경우에 대해 축 1 의 argmax 값
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()
# normal 함수 이용
steps = np.random.normal(loc=0, scale=0.25,
 size=(nwalks, nsteps))