

#케라스에 대해 알아봅시다
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
(x_train_all, y_train_all),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
# x_train = x_train/255
# x_val = x_val /255
x_train = x_train.reshape(-1,784)

x_val = x_val.reshape(-1,784)
#훈련할 가중치 변수를 선언합니다.
w= tf.Variable(tf.zeros(shape=(1)))
b= tf.Variable(tf.zeros(shape=(1)))

#경사 하강법 옵티마이저를 설정합니다
optimizer = tf.optimizers.SGD(lr = 0.01)
#에포크 횟수만큼 훈련합니다.
num_epochs = 10
for step in range(num_epochs):
    #자동 미분을 위해 연산 과정을 기록합니다.
    with tf.GradientTape() as tape:
        z_net = w*x_train +b
        z_net = tf.reshape(y_train,-1)
        sqr_errors = tf.square(y_train-z_net)
        mean_cost = tf.reduce_mean(sqr_errors)
    grads = tape.gradient(mean_cost, [w,b])
    optimizer.apply_gradients(zip(grads,[w,b]))
z_net.shape
tf.Tensor(y_train, unit8)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1))
model.complile(optimizer='sgd',loss='mse')
model.fit(x_train,y_train,epochs=10)