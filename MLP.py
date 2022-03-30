import random
from turtle import shape
import tensorflow as tf

# 导入数据集
# def data_iter():


# 下面定义初始的一些参数
batch_size = 256
num_inputs = 784
num_hiddens = 256
num_outputs = 10

# 下面定义超参数
W1 = tf.Variable(tf.random.normal(shape=(num_inputs,num_hiddens),mean=0.0,stddev=0.01),trainable=True)
b1 = tf.Variable(tf.zeros(shape=(num_hiddens,1)),trainable=True)
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens,num_outputs),mean=0.0,stddev=0.01),trainable=True)
b2 = tf.Variable(tf.zeros(shape=(num_outputs)),trainable=True)
params = [W1,b1,W2,b2]

def relu(X):
    return tf.math.maximum(X,0)

def net(X):
    H = relu(tf.matmul(X,W1) + b1)
    return tf.matmul(H,W2) + b2
