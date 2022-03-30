import random
import csv
import tensorflow as tf

def synthetic_data(path):
    with open(path,'r',encoding='utf-8') as file:
        reader = csv.reader(file)
        temp = [line for line in reader]
        X = tf.constant([list(map(float,line[0:-1])) for line in temp])
        y = tf.constant(list(map(float,[line[-1] for line in temp])))
    return tf.Variable(X,shape=X.shape),tf.Variable(y,shape=y.shape)

features,labels = synthetic_data("./Desktop/ML.csv")

num_inputs = 7
num_outputs = 1
problem_scale = 133

w = tf.Variable(tf.random.normal(shape=(7,1),mean=0,stddev=0.01),trainable=True)
b = tf.Variable(tf.zeros(1),trainable=True)

def net(X,w,b):
    return tf.matmul(X,w) + b

def loss(y_hat,y):
    return (y_hat - y) ** 2 / 2

def sgd(params,grads,lr):
    for param,grad in zip(params,grads):
        param.assign_sub(lr * grad / problem_scale)

lr = 0.00000001
num_epochs = 60

for epoch in range(num_epochs):
    for X,y in zip(features,labels):
        X = tf.reshape(X,(1,7))
        y = tf.reshape(y,(1,1))
        with tf.GradientTape() as g:
            l = loss(net(X,w,b),y)
        dw,db = g.gradient(l,[w,b])
        sgd([w,b],[dw,db],lr)
    train_l = loss(net(X,w,b),labels)
    # print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')


for X,y in zip(features,labels):
    X = tf.reshape(X,(1,7))
    y = tf.reshape(y,(1,1))
    print(net(X,w,b).value)