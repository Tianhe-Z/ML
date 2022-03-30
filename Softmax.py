from pickletools import uint8
import random
import tensorflow as tf
from IPython import display

# 下面定义一些初始参数
# 这里需要根据实际情况进行调整
batch_size = 100
num_inputs = 3
num_outputs = 2
inputs_size = 3

# 定义参数
w = tf.Variable(tf.random.normal(shape=(num_inputs,num_outputs),mean=0,stddev=0.01),trainable=True)
b = tf.Variable(tf.zeros(num_outputs))

# 下面需要导入数据，并且记录为features和labels，以供data_iter使用
def synthetic_data(num_examples,inputs_size):
    X1 = tf.zeros((num_examples,inputs_size))
    X1 += tf.random.uniform(shape=X1.shape,minval=0,maxval=5)
    X2 = tf.zeros((num_examples,inputs_size))
    X2 += tf.random.uniform(shape=X2.shape,minval=5,maxval=10)
    X = tf.concat([X1,X2],0)

    y_1 = tf.zeros((1,num_examples),dtype=tf.int32)
    y_2 = tf.ones((1,num_examples),dtype=tf.int32)
    y = tf.concat([y_1,y_2],1)[0]
    return X,y

features,labels = synthetic_data(1000,inputs_size)

# 下面定义softmax操作
def softmax(X):
    X_exp = tf.exp(X)
    partition = tf.reduce_sum(X_exp,1,keepdims=True)
    return X_exp / partition #这里运用了广播机制

# 下面定义整个线性回归的模型
def net(X):
    return softmax(tf.matmul(X,w) + b)

# 下面定义交叉熵损失函数
def cross_entropy(y_hat,y):
    # 注意boolean_mask函数的用法
    return -tf.math.log(tf.boolean_mask(y_hat,tf.one_hot(indices=y,depth=2)))

# 下面定义随机小批量数据的迭代器实现
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0,num_examples,batch_size):
        j = tf.constant(indices[i:min(i + batch_size,num_examples)])
        # 请注意，这里的y已经被我改成了适用于tf.one_hot()的独热编码格式
        yield tf.gather(features,j),tf.gather(labels,j)

# 这里实现随机梯度下降优化函数
# 注意，这里不需要实现自动求导功能，求导在训练时实现即可
def sgd(params,grads,lr,batch_size):
    for param,grad in zip(params,grads):
        param.assign_sub(lr * grad / batch_size)

# 这里是只训练一轮的函数实现
def train_epoch(net,data_iter,loss,sgd,lr):
    for X,y in data_iter(batch_size,features,labels):
        with tf.GradientTape() as tape:
            y_hat = net(X)
            l = loss(y_hat,y)
            #下面求导 
            dw,db = tape.gradient(l,[w,b])
            # 下面一步很关键，使用自定义的sgd函数，依据loss来更新params，也就是参数w和b
            sgd([w,b],[dw,db],lr,batch_size)
        # 优化完成后计算损失，这一步是非必要的
        train_l = loss(net(X),y)

# 下面定义总体的训练函数
def train(net,data_iter,loss,num_epochs,sgd,lr):
    for epoch in range(num_epochs):
        train_epoch(net,data_iter,loss,sgd,lr)

# 下面是训练的主题
lr = 0.01
num_epochs = 100

# print(features)
# print(labels)

train(net,data_iter,cross_entropy,num_epochs,sgd,lr)

# print(w)
# print(b)

for X,y in data_iter(batch_size,features,labels):
    print(net(X))