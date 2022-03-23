import tensorflow as tf
import random

# 定义输入输出规模
# 在测试代码时由于使用了随机生成的数据，因此没有用到nun_inputs和num_outputs
# num_inputs = 
# num_outputs = 
inputs_size = 2

# 定义随机生成数据的函数
def synthetic_data(w,b,num_examples):
    X = tf.zeros((num_examples,inputs_size))
    X += tf.random.normal(shape=X.shape)
    # 注意下一步的reshape，需要将w从定义时的行向量转变为列向量，否则将违反矩阵乘法法则
    y = tf.matmul(X,tf.reshape(w,(-1,1))) + b

    # 经过检测，这里使用X与w相乘之后的y就是一个列向量，无需reshape，可能只是为了保险
    y = tf.reshape(y,(-1,1))
    return X,y

true_w = tf.constant([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

# 定义实现随机小批量读取数据的函数
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    # 下面的循环需要好好理解
    for i in range(0,num_examples,batch_size):
        # indices里面的就是对列表用下表截取
        j = tf.constant(indices[i:min(i + batch_size,num_examples)])
        yield tf.gather(features,j),tf.gather(labels,j)

# 下面是初始化参数
w = tf.Variable(tf.random.normal(shape=(2,1),mean=0,stddev=0.01),trainable=True)
b = tf.Variable(tf.zeros(1),trainable=True)

# 下面定义模型
def linreg(X,w,b):
    return tf.matmul(X,w) + b

# 定义损失函数
def squared_loss(y_hat,y):
    return (y_hat - tf.reshape(y,y_hat.shape)) ** 2 / 2

# 实现梯度下降更新参数
# 这里没有求导功能，需要输入梯度
def sgd(params,grads,lr,batch_size):
    for param,grad in zip(params,grads):
        param.assign_sub(lr * grad / batch_size)
        # 或者也可以写成下面的形式
        # tf.assign_sub(param,lr * grad / batch_size)

# 定义超参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

# 循环训练
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        with tf.GradientTape() as g:
            l = loss(net(X,w,b),y)
        # 下面计算梯度
        dw,db = g.gradient(l,[w,b])
        # 下面使用优化函数对参数进行优化
        sgd([w,b],[dw,db],lr,batch_size)
    # 优化完成后计算损失
    train_l = loss(net(features,w,b),labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

print(true_w,true_b)
print(w,b)