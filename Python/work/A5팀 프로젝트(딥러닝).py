
from SyncRNG import SyncRNG
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
pd.set_option('display.max_columns', None) ## 모든 열을 출력한다
# https://tensorflow.blog/%EC%BC%80%EB%9D%BC%EC%8A%A4-%EB%94%A5%EB%9F%AC%EB%8B%9D/3-6-%EC%A3%BC%ED%83%9D-%EA%B0%80%EA%B2%A9-%EC%98%88%EC%B8%A1-%ED%9A%8C%EA%B7%80-%EB%AC%B8%EC%A0%9C/

# 데이터 불러오기
raw_data = pd.read_csv('E:/GoogleDrive/포트폴리오/A5팀 R과 Python기반 머신러닝과 딥러닝 분석 비교(12월22일)/dataset/product.csv',encoding='cp949')

# 데이터 셋  7:3 으로 분할
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]

# R에서는 데이터프레임이 1부터 시작하기 때문에
# python에서 0행과 R에서 1행이 같은 원리로
# 같은 인덱스 번호를 가진다면 -1을 해주어 같은 데이터를 가지고 오게 한다.
# 인덱스 수정-R이랑 같은 데이터 가져오려고
for i in range(0,len(idx)):
    idx[i]=idx[i]-1

# 학습데이터, 테스트데이터 생성
train=raw_data.loc[idx] # 70%
#train=train.sort_index(ascending=True)
test=raw_data.drop(idx) # 30%

x_train = train.제품_적절성
y_train = train.제품_만족도
x_test = test.제품_적절성
y_test = test.제품_만족도

class Neuron:
    def __init__(self):
        self.w = 1  # 가중치를 초기화합니다
        self.b = 1 # 절편을 초기화합니다

    def forpass(self, x):
        y_hat = x * self.w + self.b  # 직선 방정식을 계산합니다
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err  # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err  # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def fit(self, x, y, lr, epochs=400):
        for i in range(epochs):  # 에포크만큼 반복합니다
            for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복합니다
                n=len(x)
                y_hat = self.forpass(x_i)  # 정방향 계산
                err = -(2/n)*(y_i - y_hat)  # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err)  # 역방향 계산
                self.w -= w_grad*lr  # 가중치 업데이트
                self.b -= b_grad*lr # 절편 업데이트
            if i % 10 == 0:
                print('[',i,']',err, self.w, self.b)

neuron = Neuron()
neuron.fit(x_train, y_train, 0.01)


predict=[]
predict = x_test * neuron.w + neuron.b
er=(predict-y_test)**2 ##오차제곱
mse=er.mean(axis=0) #mse값
rmse=np.sqrt(mse)  #rmse 값
print(rmse)
print(neuron.w, neuron.b)
print(1-(er.sum()/((predict-x_test.mean())**2).sum()))
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(predict, y_test)))

plt.scatter(x_train, y_train, label='true')
plt.scatter(x_test, predict, label='pred')
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (5, 5 * neuron.w + neuron.b)
# plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],color='orange')
plt.plot(x_test, predict, 'r-', color='yellow')
plt.legend()
plt.show

# help(keras.Sequential.compile)
# help(tf.keras.optimizers)
# model = keras.Sequential()
# model.compile(optimizer='adam',loss='mse',metrics=['mse'])
# history = model.fit(x_train, y_train, epochs = 400, validation_data=(x_test,y_test))
#
# plt.plot(history.history['mse'])
# plt.plot(history.history['val_mse'])
# plt.show
# ####################################################################
# #https://limitsinx.tistory.com/31
# # 데이터 불러오기
# import tensorflow.compat.v1 as tf
# raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/product.csv',encoding='cp949')
# tf.disable_v2_behavior()
#
# # 데이터 셋  7:3 으로 분할
# v=list(range(1,len(raw_data)+1))
# s=SyncRNG(seed=42)
# ord=s.shuffle(v)
# idx=ord[:round(len(raw_data)*0.7)]
#
# # R에서는 데이터프레임이 1부터 시작하기 때문에
# # python에서 0행과 R에서 1행이 같은 원리로
# # 같은 인덱스 번호를 가진다면 -1을 해주어 같은 데이터를 가지고 오게 한다.
# # 인덱스 수정-R이랑 같은 데이터 가져오려고
# for i in range(0,len(idx)):
#     idx[i]=idx[i]-1
#
# # 학습데이터, 테스트데이터 생성
# train=raw_data.loc[idx] # 70%
# #train=train.sort_index(ascending=True)
# test=raw_data.drop(idx) # 30%
#
# x_train = np.array(train.제품_적절성, dtype=np.float32)
# y_train = np.array(train.제품_만족도, dtype=np.float32)
# x_test = np.array(test.제품_적절성, dtype=np.float32)
# y_test = np.array(test.제품_만족도, dtype=np.float32)
#
# W = tf.Variable(tf.random_normal([1]), name='weight')
#
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = x_train * W + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# for step in range(10001):
#
#     sess.run(train)
#
#     if step % 100 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))
#오차는 0.2695081 , W=0.7606595 , b= 0.717101

# #######################딥러닝
# raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/product.csv',encoding='cp949')
# x_data = np.array(raw_data.iloc[:, 1], dtype=np.float32)
# y_data = np.array(raw_data.iloc[:, 2], dtype=np.float32)
#
# y_data = y_data.reshape((264, 1))
#
# # import tensorflow.compat.v1 as tf2
# # tf2.disable_v2_behavior()
# # X = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
# # Y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')
# #
# #
# # W = tf.Variable(tf.random_normal([2, 1]), name='weight')
# # b = tf.Variable(tf.random_normal([1]), name='bias')
# #
# # hypothesis = tf.matmul(X, W) + b
#
# # Keras의 Sequential모델을 선언
# model = keras.Sequential([
#     # 첫 번째 Layer: 데이터를 신경망에 집어넣기
#     keras.layers.Dense(32, activation=tf.nn.leaky_relu, input_shape = (1, )),
#     # 두번째 층
#     keras.layers.Dense(32, activation=tf.nn.leaky_relu),
#     # 세번째 출력층: 예측 값 출력하기
#     keras.layers.Dense(1)
# ])
#
# help(keras.Sequential.compile)
# # 모델을 학습시킬 최적화 방법, loss 계산 방법, 평가 방법 설정
# model.compile(optimizer='adam',loss='mse',metrics=['mse', 'binary_crossentropy'])
# # 모델 학습
# history = model.fit(x_data,y_data, epochs = 100, batch_size = 100)
#
# # 결과를 그래프로 시각화
# plt.scatter(x_data, y_data, label='y_true')
# plt.scatter(x_data, model.predict(x_data), label='y_pred')
# # plt.scatter(y_data, model.predict(x_data), label='y_pred')
# plt.legend()
# plt.savefig("plot.png")
#
#
#
#
# # https://blog.naver.com/PostView.naver?blogId=shino1025&logNo=221600585667&parentCategoryNo=&categoryNo=45&viewDate=&isShowPopularPosts=true&from=search
#
#
# ###########################
# #https://wikidocs.net/111472
#
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
# import tensorflow as tf1
# tf1.compat.v1.enable_eager_execution()
# raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/product.csv',encoding='cp949')
# x = tf1.convert_to_tensor(np.array(raw_data.제품_적절성, dtype=np.float64))
#
# y = tf1.convert_to_tensor(np.array(raw_data.제품_만족도, dtype=np.float64))
#
# model = Sequential()
#
# # 출력 y의 차원은 1. 입력 x의 차원(input_dim)은 1
# # 선형 회귀이므로 activation은 'linear'
# model.add(Dense(1, input_dim=1, activation='linear'))
#
# # sgd는 경사 하강법을 의미. 학습률(learning rate, lr)은 0.01.
# sgd = optimizers.SGD(lr=0.01)
#
# # 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
# model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
#
# # 주어진 x와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.
# model.fit(x, y, epochs=300)
#
# plt.plot(x, model.predict(x), 'b', x, y, 'k.')
#
# print(model.predict([12]))
# ########################################
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# # tape_gradient()는 자동 미분 기능을 수행합니다. 임의로 라는 식을 세워보고, 에 대해 미분해보겠습니다.
#
# w = tf.Variable(2.)
#
# def f(w):
#   y = w**2
#   z = 2*y + 5
#   return z
# # gradients를 출력하면 에 대해 미분한 값이 저장된 것을 확인할 수 있습니다.
#
# with tf.GradientTape() as tape:
#   z = f(w)
#
# gradients = tape.gradient(z, [w])
# print(gradients)
# # [<tf.Tensor: shape=(), dtype=float32, numpy=8.0>]
# # 이 자동 미분 기능을 사용하여 선형 회귀를 구현해봅시다.
# # 학습될 가중치 변수를 선언
#
# w = tf.Variable(4.0)
# b = tf.Variable(1.0)
# # 가설을 함수로서 정의합니다.
#
# @tf.function
# def hypothesis(x):
#   return w*x + b
# # 현재의 가설에서 w와 b는 각각 4와 1이므로 임의의 입력값을 넣었을 때의 결과는 다음과 같습니다.
#
#
# print(hypothesis(x).numpy())
# # [15. 21. 23. 25.]
# # 다음과 같이 평균 제곱 오차를 손실 함수로서 정의합니다.
#
# @tf.function
# def mse_loss(y_pred, y):
#   # 두 개의 차이값을 제곱을 해서 평균을 취한다.
#   return tf.reduce_mean(tf.square(y_pred - y))
#
#
# raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/product.csv',encoding='cp949')
# x = np.array(raw_data.제품_적절성, dtype=np.float32)
# y = np.array(raw_data.제품_만족도, dtype=np.float32)
# # 옵티마이저는 경사 하강법을 사용하되, 학습률(learning rate)는 0.01을 사용합니다.
#
# optimizer = tf.optimizers.SGD(0.01)
# # 약 300번에 걸쳐서 경사 하강법을 수행하겠습니다.
#
# for i in range(3001):
#   with tf.GradientTape() as tape:
#     # 현재 파라미터에 기반한 입력 x에 대한 예측값을 y_pred
#     y_pred = hypothesis(x)
#
#     # 평균 제곱 오차를 계산
#     cost = mse_loss(y_pred, y)
#
#   # 손실 함수에 대한 파라미터의 미분값 계산
#   gradients = tape.gradient(cost, [w, b])
#
#   # 파라미터 업데이트
#   optimizer.apply_gradients(zip(gradients, [w, b]))
#
#   if i % 100 == 0:
#     print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))
# # epoch :   0 | w의 값 : 8.2133 | b의 값 : 1.664 | cost : 1402.555542
# # ... 중략 ...
# # epoch : 280 | w의 값 : 10.6221 | b의 값 : 1.191 | cost : 1.091434
# # epoch : 290 | w의 값 : 10.6245 | b의 값 : 1.176 | cost : 1.088940
# # epoch : 300 | w의 값 : 10.6269 | b의 값 : 1.161 | cost : 1.086645
# # w와 b값이 계속 업데이트 됨에 따라서 cost가 지속적으로 줄어드는 것을 확인할 수 있습니다. 학습된 w와 b의 값에 대해서 임의 입력을 넣었을 경우의 예측값을 확인해봅시다.
#
# x_test = [3.5, 5, 5.5, 6]
# print(hypothesis(x_test).numpy())
# # [38.35479  54.295143 59.608593 64.92204 ]
#

##########################################
#2번
#################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from SyncRNG import SyncRNG

raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/weather.csv',encoding='cp949')
raw_data.head()
#날짜 제거
raw_data.drop(['Date'],axis=1,inplace=True)

#범주형변수 레이블인코더
le = LabelEncoder()
cat_cols = ['WindGustDir', 'WindDir', 'RainToday','RainTomorrow']
raw_data[cat_cols] = raw_data[cat_cols].apply(le.fit_transform)

#각 변수간 범위가 전부 다르기 때문에 최소-최대 정규화로 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Sunshine', 'WindGustDir',
       'WindGustSpeed', 'WindDir', 'WindSpeed', 'Humidity', 'Pressure',
       'Cloud', 'Temp', 'RainToday', 'RainTomorrow']
raw_data[cols] = scaler.fit_transform(raw_data[cols])


# 데이터 셋  7:3 으로 분할
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]

# R에서는 데이터프레임이 1부터 시작하기 때문에
# python에서 0행과 R에서 1행이 같은 원리로
# 같은 인덱스 번호를 가진다면 -1을 해주어 같은 데이터를 가지고 오게 한다.
# 인덱스 수정-R이랑 같은 데이터 가져오려고
for i in range(0,len(idx)):
    idx[i]=idx[i]-1

# 학습데이터, 테스트데이터 생성
train=raw_data.loc[idx] # 70%
#train=train.sort_index(ascending=True)
test=raw_data.drop(idx) # 30%

#결측치 제거
# raw_data=raw_data.dropna()
train=raw_data.dropna()
test=raw_data.dropna()
#np.sum(train.isnull(),axis=0)


x_train = np.array(train.iloc[:,0:13], dtype=np.float32)
y_train = np.array(train.iloc[:,-1], dtype=np.float32)
x_test = np.array(test.iloc[:,0:13], dtype=np.float32)
y_test = np.array(test.iloc[:,-1], dtype=np.float32)

x_train.shape, x_test.shape, y_train.shape,y_test.shape

#
# class LogisticNeuron:
#
#     def __init__(self):
#         self.w = None
#         self.b = None
#
#     def forpass(self, x):
#         z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
#         return z
#
#     def backprop(self, x, err):
#         w_grad = x * err  # 가중치에 대한 그래디언트를 계산합니다
#         b_grad = 1 * err  # 절편에 대한 그래디언트를 계산합니다
#         return w_grad, b_grad
#
#     def activation(self, z):
#         z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
#         a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
#         return a
#
#     def fit(self, x, y, epochs=100):
#         self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
#         self.b = 0  # 절편을 초기화합니다.
#         for i in range(epochs):  # epochs만큼 반복합니다
#             for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복합니다
#                 n=len(x)
#                 z = self.forpass(x_i)  # 정방향 계산
#                 a = self.activation(z)  # 활성화 함수 적용
#                 err = -(2/n)*(y_i - a)  # 오차 계산
#                 w_grad, b_grad = self.backprop(x_i, err)  # 역방향 계산
#                 self.w -= w_grad  # 가중치 업데이트
#                 self.b -= b_grad  # 절편 업데이트
#             if i % 20 == 0:
#                 print('[',i,']',err, self.w, self.b)
#
#     def predict(self, x):
#         z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
#         a = self.activation(np.array(z))  # 활성화 함수 적용
#         return a > 0.5
#
# neuron = LogisticNeuron()
# neuron.fit(x_train, y_train)
#
# np.mean(neuron.predict(x_test) == y_test)

class SingleLayer:

    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x, err):
        w_grad = x * err  # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err  # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def fit(self, x, y,lr, epochs=401):
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.

        for j in range(epochs):  # epochs만큼 반복합니다
            loss = 0
            # 인덱스를 섞습니다
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:  # 모든 샘플에 대해 반복합니다
                n=len(x)
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(2/n)*(y[i] - a)  # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err)  # 역방향 계산

                self.w -= w_grad*lr  # 가중치 업데이트
                self.b -= b_grad*lr # 절편 업데이트
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다
                a = np.clip(a, 1e-10, 1 - 1e-10)
                loss += -(y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
            # 에포크마다 평균 손실을 저장합니다
            self.losses.append(loss / len(y))
            if j % 100 == 0:
             print('[',j,']',err,self.b,self.w)

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
        return np.array(z) > 0  # 스텝 함수 적용

    def proba(self,x):  #ROC 커브용 확률출력 메서드
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

layer = SingleLayer()
layer.fit(x_train, y_train, 1)
print(layer.score(x_test, y_test))
layer.predict(x_test)
from sklearn.linear_model import LogisticRegression
#print(inspect.getsource(LogisticRegression))
log=LogisticRegression()
log.fit(x_train, y_train)
print(log.score(x_test, y_test))

plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

from sklearn.metrics import roc_auc_score
print('ROC auc value : {}'.format(roc_auc_score(y_test,layer.predict(x_test)))) # auc에 대한 면적
#ROC auc value : 0.7523388773388773

from sklearn.metrics import classification_report
print(classification_report(layer.predict(x_test) , y_test))
#               precision    recall  f1-score   support
#        False       0.97      0.91      0.93       316
#         True       0.54      0.78      0.64        45
#     accuracy                           0.89       361
#    macro avg       0.75      0.84      0.79       361
# weighted avg       0.91      0.89      0.90       361



#시각화
##ROC커브 시각화
pro=layer.proba(x_test)
from sklearn.metrics import roc_curve
import inspect

fprs, tprs, thresholds = roc_curve(y_test, pro,pos_label=1)
precisions, recalls, thresholds = roc_curve(y_test, pro,pos_label=1)

# ROC
plt.figure(figsize=(5,5))
plt.plot(fprs,tprs,label='ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


####################
# from sklearn.linear_model import  SGDClassifier
# sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)
#
# sgd.fit(x_train,y_train)
# sgd.score(x_test,y_test)
# a=sgd.predict(x_test)
#
# from sklearn.metrics import roc_curve
# fprs, tprs, thresholds = roc_curve(y_test, a)
# precisions, recalls, thresholds = roc_curve(y_test, a)
#
#
# plt.figure(figsize=(15,5))
# # # 대각선
# # plt.plot([0,1],[0,1],label='STR')
# # ROC
# plt.plot(fprs,tprs,label='ROC')
#
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend()
# plt.grid()
# plt.show()
#
# from sklearn.metrics import roc_auc_score
# print('roc auc value {}'.format(roc_auc_score(y_test,a))) # 이  value는 auc에 대한 면적을 나타낸 것이다.

# model = Sequential()
# model.add(Dense(1, input_dim=1, activation='sigmoid'))
#
# sgd = optimizers.SGD(lr=0.01)
#
# import tensorflow as tf1
# tf1.enable_eager_execution()
# model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
#
# model.fit(x, y, epochs=200)
# plt.plot(x, model.predict(x), 'b', x,y, 'k.')
# print(model.predict([1, 2, 3, 4, 4.5]))
# print(model.predict([11, 21, 31, 41, 500]))

#################################################

#
#
# import pandas as pd
# from tensorflow import keras
# import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
#
# #https://limitsinx.tistory.com/35
# # 데이터 불러오기
# raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/weather.csv',encoding='cp949')
# tf.disable_v2_behavior()
#
# raw_data.drop(['Date','RainToday','WindGustDir','WindDir'],axis=1,inplace=True)
# raw_data.info()
# encoder = LabelEncoder()
# encoder.fit(raw_data.iloc[:,-1])
# raw_data.iloc[:,-1] = encoder.transform(raw_data.iloc[:,-1])
# x_train.shape
# x_train = np.array(raw_data.iloc[:,0:10], dtype=np.float64)
# y_train = np.array(raw_data.iloc[:,-1], dtype=np.float64)
# # placeholder을 만들때는 shape에 주의
#
# X = tf.placeholder(tf.float32, shape=[None, 10])
#
# Y = tf.placeholder(tf.float32, shape=[366, ])
#
# W = tf.Variable(tf.random_normal([10, 1]), name='weight')
#
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# # hypothesis를 sigmoid에 통과 : 0~1의 값으로 나오게 될것임.
#
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# # sigmoid를 통과한 값이 0.5보다 크면 1 아니면 0 으로 기준 설정(Cast)
#
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # True = 1, False = 0
#
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())  # Variable 초기화
#
#
# for step in range(5001):
#
#     cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
#
#     if step % 1000 == 0:
#         print("%05d" % step, cost_val)
#
# h, c, a, w, b, cost = sess.run([hypothesis, predicted, accuracy, W, b, cost],
#
#                                feed_dict={X: x_data, Y: y_data})
#
# print("\n [5001번 학습결과]")
#
# # print(h)
#
# print("1. 시그모이드 적용 : ", h.T)
#
# print("2. + 활성함수 적용 : ", [int(x) for x in c])
#
# print("3. 기존 정답과 비교 : ", sum(y_data, []))
#
# print("정확도(Accuracy) : ", a)
#
# print("Weighting : ", w)
#
# print("bias :", b)
#
# print("cost : ", cost)

################################################################################
#3번
from SyncRNG import SyncRNG
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

class MultiClassNetwork:

    def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
        self.units = units  # 은닉층의 뉴런 개수
        self.batch_size = batch_size  # 배치 크기
        self.w1 = None  # 은닉층의 가중치
        self.b1 = None  # 은닉층의 절편
        self.w2 = None  # 출력층의 가중치
        self.b2 = None  # 출력층의 절편
        self.a1 = None  # 은닉층의 활성화 출력
        self.losses = []  # 훈련 손실
        self.val_losses = []  # 검증 손실
        self.lr = learning_rate  # 학습률
        self.l1 = l1  # L1 손실 하이퍼파라미터
        self.l2 = l2  # L2 손실 하이퍼파라미터

    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1  # 첫 번째 층의 선형 식을 계산합니다
        self.a1 = self.sigmoid(z1)  # 활성화 함수를 적용합니다
        z2 = np.dot(self.a1, self.w2) + self.b2  # 두 번째 층의 선형 식을 계산합니다.
        return z2

    def backprop(self, x, err):
        m = len(x)  # 샘플 개수
        # 출력층의 가중치와 절편에 대한 그래디언트를 계산합니다.
        w2_grad = np.dot(self.a1.T, err) / m
        b2_grad = np.sum(err) / m
        # 시그모이드 함수까지 그래디언트를 계산합니다.
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        # 은닉층의 가중치와 절편에 대한 그래디언트를 계산합니다.
        w1_grad = np.dot(x.T, err_to_hidden) / m
        b1_grad = np.sum(err_to_hidden, axis=0) / m
        return w1_grad, b1_grad, w2_grad, b2_grad

    def sigmoid(self, z):
        z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def softmax(self, z):
        # 소프트맥스 함수
        z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)

    def init_weights(self, n_features, n_classes):
        self.w1 = np.random.normal(0, 1,(n_features, self.units))  # (특성 개수, 은닉층의 크기)
        self.b1 = np.zeros(self.units)  # 은닉층의 크기
        self.w2 = np.random.normal(0, 1,(self.units, n_classes))  # (은닉층의 크기, 클래스 개수)
        self.b2 = np.zeros(n_classes)

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        np.random.seed(42)
        self.init_weights(x.shape[1], y.shape[1])  # 은닉층과 출력층의 가중치를 초기화합니다.
        # epochs만큼 반복합니다.
        for i in range(epochs):
            loss = 0
            print('.', end='')
            # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
            for x_batch, y_batch in self.gen_batch(x, y):
                a = self.training(x_batch, y_batch)
                # 안전한 로그 계산을 위해 클리핑합니다.
                a = np.clip(a, 1e-10, 1 - 1e-10)
                # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
                loss += np.sum(-y_batch * np.log(a))
            self.losses.append((loss + self.reg_loss()) / len(x))
            # 검증 세트에 대한 손실을 계산합니다.
            self.update_val_loss(x_val, y_val)

    # 미니배치 제너레이터 함수
    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size  # 미니배치 횟수
        if length % self.batch_size:
            bins += 1  # 나누어 떨어지지 않을 때
        indexes = np.random.permutation(np.arange(len(x)))  # 인덱스를 섞습니다.
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]  # batch_size만큼 슬라이싱하여 반환합니다.

    def training(self, x, y):
        m = len(x)  # 샘플 개수를 저장합니다.
        z = self.forpass(x)  # 정방향 계산을 수행합니다.
        a = self.softmax(z)  # 활성화 함수를 적용합니다.
        err = -(y - a)  # 오차를 계산합니다.
        # 오차를 역전파하여 그래디언트를 계산합니다.
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
        # 그래디언트에서 페널티 항의 미분 값을 뺍니다
        w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
        w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m
        # 은닉층의 가중치와 절편을 업데이트합니다.
        self.w1 -= self.lr * w1_grad
        self.b1 -= self.lr * b1_grad
        # 출력층의 가중치와 절편을 업데이트합니다.
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a

    def predict(self, x):
        z = self.forpass(x)  # 정방향 계산을 수행합니다.
        return np.argmax(z, axis=1)  # 가장 큰 값의 인덱스를 반환합니다.

    def score(self, x, y):
        # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
        return np.mean(self.predict(x) == np.argmax(y, axis=1))

    def reg_loss(self):
        # 은닉층과 출력층의 가중치에 규제를 적용합니다.
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + \
               self.l2 / 2 * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))

    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)  # 정방향 계산을 수행합니다.
        a = self.softmax(z)  # 활성화 함수를 적용합니다.
        a = np.clip(a, 1e-10, 1 - 1e-10)  # 출력 값을 클리핑합니다.
        # 크로스 엔트로피 손실과 규제 손실을 더하여 리스트에 추가합니다.
        val_loss = np.sum(-y_val * np.log(a))
        self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))


# 데이터 불러오기
raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/iris.csv',encoding='cp949')

# 데이터 셋  7:3 으로 분할
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=38)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]

# R에서는 데이터프레임이 1부터 시작하기 때문에
# python에서 0행과 R에서 1행이 같은 원리로
# 같은 인덱스 번호를 가진다면 -1을 해주어 같은 데이터를 가지고 오게 한다.
# 인덱스 수정-R이랑 같은 데이터 가져오려고
for i in range(0,len(idx)):
    idx[i]=idx[i]-1

# 학습데이터, 테스트데이터 생성
train=raw_data.loc[idx] # 70%
#train=train.sort_index(ascending=True)
test=raw_data.drop(idx) # 30%

x_train = np.array(train.iloc[:,0:4], dtype=np.float32)
y_train = np.array(train.Species)
x_test = np.array(test.iloc[:,0:4], dtype=np.float32)
y_test = np.array(test.Species)

####x_val = x_test
####y_val = y_test

#라벨인코더
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test =  le.fit_transform(y_test)

tf.keras.utils.to_categorical([0,1,2])
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)
#차원 ((105, 3), (45, 3))
y_train_encoded.shape, y_test_encoded.shape

#class 이름
class_name=['setosa','versicolor','virginica']

#빈도수 확인
raw_data.Species.unique()
np.bincount(y_test)
np.bincount(y_train)

#차원 (105,4) , (45,4)
x_train.shape
x_test.shape

#multiclassnetwork으로 다중분류 신경망 훈련하기
fc=MultiClassNetwork(units=3, batch_size=1)
fc.fit(x_train,y_train_encoded, x_val=x_test,y_val=y_test_encoded,epochs=200)

plt.plot(fc.losses)
plt.plot(fc.val_losses)
plt.legend(['train_loss','val_loss'])
plt.show
fc.score(x_test,y_test_encoded)
# 분류 정확도:0.9777777777777777
fc.predict(x_test)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
#        1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2], dtype=int64)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()

#https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object
model.add(Dense(4, activation='sigmoid',input_shape=(4,))) #은닉층 갯수 4개, input_shape는 train feature수
model.add(Dense(3, activation='softmax')) #출력층 갯수 3개 답의 class갯수와 같다.
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train_encoded, epochs=40,validation_data=(x_test,y_test_encoded))
model.evaluate(x_test,y_test_encoded)
#정확도 0.9555555582046509
model.predict(x_test)

# array([[0.4522499 , 0.31179902, 0.23595104],
#        [0.44566822, 0.30944934, 0.24488243],
#        [0.44525114, 0.32209396, 0.23265482],
#        [0.42326656, 0.33701238, 0.23972115],
#        [0.4281756 , 0.34606394, 0.2257605 ],
#        [0.44037837, 0.32445076, 0.23517092],
#        [0.42748642, 0.32706413, 0.24544944],
#        [0.4284664 , 0.31784222, 0.25369132],
#        [0.44067314, 0.3127521 , 0.24657476],
#        [0.42895272, 0.33032516, 0.24072213],
#        [0.44122535, 0.31444526, 0.24432935],
#        [0.44126   , 0.3284938 , 0.23024617],
#        [0.45614907, 0.30411214, 0.23973875],
#        [0.44909722, 0.2977419 , 0.25316083],
#        [0.41977122, 0.3316457 , 0.24858315],
#        [0.27917764, 0.34956533, 0.3712571 ],
#        [0.31540644, 0.3188664 , 0.3657271 ],
#        [0.28867584, 0.3410459 , 0.37027818],
#        [0.35047838, 0.3125794 , 0.33694223],
#        [0.34149104, 0.30504796, 0.35346097],
#        [0.31795797, 0.32699043, 0.35505167],
#        [0.29209304, 0.32545182, 0.38245517],
#        [0.3227658 , 0.32290334, 0.35433087],
#        [0.2827632 , 0.339929  , 0.37730783],
#        [0.30166587, 0.34221452, 0.35611963],
#        [0.28371364, 0.34234285, 0.37394348],
#        [0.2740027 , 0.3454548 , 0.38054246],
#        [0.3319054 , 0.31969517, 0.34839946],
#        [0.31614813, 0.32849467, 0.35535723],
#        [0.3030603 , 0.3390534 , 0.35788634],
#        [0.24539818, 0.3429699 , 0.4116319 ],
#        [0.24868016, 0.33567354, 0.4156463 ],
#        [0.23657829, 0.35610837, 0.4073134 ],
#        [0.26436466, 0.33271018, 0.40292513],
#        [0.27320746, 0.32198048, 0.40481207],
#        [0.2615724 , 0.3366646 , 0.401763  ],
#        [0.22904553, 0.33506966, 0.43588483],
#        [0.24771193, 0.3444542 , 0.40783384],
#        [0.27962574, 0.33704326, 0.38333097],
#        [0.24291396, 0.3390189 , 0.4180671 ],
#        [0.26737437, 0.31716862, 0.415457  ],
#        [0.2364627 , 0.35087302, 0.41266426],
#        [0.28143716, 0.33769146, 0.38087144],
#        [0.24918883, 0.34547296, 0.40533826],
#        [0.25921148, 0.3484995 , 0.3922891 ]], dtype=float32)


print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.show


#######################################
#3번 케라스 써서
import os
import tensorflow.compat.v1 as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#tf.debugging.set_log_device_placement(True)
from SyncRNG import SyncRNG
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
raw_data = pd.read_csv('E:/GoogleDrive/절대삭제노노/포트폴리오/A5팀 R과 Python기반 머신러닝과 딥러닝 분석 비교(12월22일)/dataset/iris.csv')
class_names = ['Setosa','Versicolor','Virginica']

import seaborn as sns
sns.pairplot(raw_data, hue="Species", size=2, diag_kind="kde")


# 데이터 셋  7:3 으로 분할
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=38)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]

# R에서는 데이터프레임이 1부터 시작하기 때문에
# python에서 0행과 R에서 1행이 같은 원리로
# 같은 인덱스 번호를 가진다면 -1을 해주어 같은 데이터를 가지고 오게 한다.
# 인덱스 수정-R이랑 같은 데이터 가져오려고
for i in range(0,len(idx)):
    idx[i]=idx[i]-1

# 학습데이터, 테스트데이터 생성
train=raw_data.loc[idx] # 70%
#train=train.sort_index(ascending=True)
test=raw_data.drop(idx) # 30%

x_train = np.array(train.iloc[:,0:4], dtype=np.float32)
y_train = np.array(train.Species.replace(['setosa','versicolor','virginica'],[0,1,2]))
x_test = np.array(test.iloc[:,0:4], dtype=np.float32)
y_test = np.array(test.Species.replace(['setosa','versicolor','virginica'],[0,1,2]))

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train_encoded, epochs=100, batch_size=1, validation_data=(x_test, y_test_encoded))

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'pred'], loc='upper left')
plt.show()
print("\n 테스트 loss: %.4f" % (model.evaluate(x_test, y_test_encoded)[0]))
print("\n 테스트 accuracy: %.4f" % (model.evaluate(x_test, y_test_encoded)[1]))


import matplotlib.pyplot as plt
history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'ro', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # 그림을 초기화합니다
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'y', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




###예측 시각화
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(3),class_names,rotation=15)

  thisplot = plt.bar(range(3), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  title_font = {
      'fontsize': 10,
  }
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'


  plt.title("{}. {} {:2.0f}% ({})".format(i,class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color,fontdict=title_font)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')




predictions=model.predict(x_test)

####0~24
num_rows = 5
num_cols = 5
num_table = num_rows*num_cols
plt.figure(figsize=(3*num_cols, 3*num_rows))
for i in range(num_table):
    plt.subplot(num_rows, num_cols,i+1)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()


####25~44
num_rows = 4
num_cols = 5
num_table = num_rows*num_cols
plt.figure(figsize=(3*num_cols, 3*num_rows))
for i in range(num_table):
    plt.subplot(num_rows, num_cols,i+1)
    plot_value_array(i+25, predictions[i+25], y_test)
    if i>=19:
        break
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report
result=[np.argmax(x) for x in predictions]
print(classification_report( result, y_test))


num_rows = 1
num_cols = 1
num_table = num_rows*num_cols
plt.figure(figsize=(5*num_cols, 5*num_rows))

plot_value_array(26,predictions[26], y_test)
plt.tight_layout()
plt.show()
predictions[23]
