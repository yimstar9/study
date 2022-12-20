from SyncRNG import SyncRNG
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None) ## 모든 열을 출력한다

# 데이터 불러오기
raw_data = pd.read_csv('E:/GoogleDrive/절대삭제노노/포트폴리오/A5팀 R과 Python기반 머신러닝과 딥러닝 분석 비교(12월22일)/dataset/product.csv',encoding='cp949')

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
er=(y_test-predict)**2 ##오차제곱
se=(predict-(sum(y_test)/len(y_test)))**2
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


y_train_mean=round(sum(y_train)/len(y_train),2)
SST = sum((y_train-y_train_mean)**2)

from sklearn.metrics import r2_score
print(r2_score(y_test,predict))
SSR=sum(er)
SSE=sum(se)
SSR/(SSR+SSE)
SSE/(SSR+SSE)
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




#######################################
#3번
#####################################
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

#데이터 시각화
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
