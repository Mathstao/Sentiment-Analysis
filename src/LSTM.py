# -*- coding: utf-8 -*-
import math
import sys
import numpy as np
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.contrib import rnn
from sklearn.metrics import confusion_matrix


# 模型参数
learning_rate = 0.001  # 学习率
train = 1  # 训练/测试
batch_size = 256  # 批大小
max_epoch = 50  # 最大迭代次数
display_step = 1  # 模型保存及打印间隔
keep_prob = 0.8  # 训练时神经元的保存概率（防止过拟合）
embedding_dim = 200  # 词向量维度
max_L = 40  # 句子统一长度
max_words = 30000  # 只取词频数最高的若干个词
n_hidden = 256  # LSTM中神经元的维度
n_classes = 2  # 类别数量
class_names = ["positive", "negative"]  # 类别名称
padding_char = 0  # 句子填充的标记
start_char = 0  # 句子开始的标记
oov_char = 0  # 句子词语替换的标记

curr_path = sys.path[0]

# 读取数据
data_train = np.load(curr_path + './../data/train_data.npy')
data_test = np.load(curr_path + './../data/test_data.npy')
x_train = data_train[:, 0]
temp = data_train[:, 1]
y_train = np.zeros([len(temp), 2])
for i in range(len(temp)):
    y_train[i, :] = temp[i]

x_test = data_test[:, 0]
temp = data_test[:, 1]
y_test = np.zeros([len(temp), 2])
for i in range(len(temp)):
    y_test[i, :] = temp[i]

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))
print("load data sucessfully!")


# 读取词向量矩阵
def get_new_w2v(w2v, max_words, embedding_dim):
    if np.shape(w2v)[1] != embedding_dim:
        print('The shape of w2v is not corresponding with the embedding dimension!')
        return None
    elif (np.shape(w2v)[0] + 3) < max_words:
        print('Max_words is too large!')
        return None
    else:
        return w2v[:max_words, :]


# 得到在一次迭代中各批训练数据的下标
def next_batch_index(num, batch_size):
    k = math.ceil(num / batch_size)
    res = batch_size * k - num
    a = np.zeros((k, batch_size))
    perm = np.random.permutation(num)
    for i in range(k - 1):
        a[i, :] = perm[batch_size * i:batch_size * (i + 1)]
    temp = perm[-res:]
    if res == 0:
        a[k - 1, :] = perm[-batch_size:]
    elif res != 0:
        while len(temp) < batch_size:
            temp = np.append(temp, np.random.randint(0, num - 1))
            a[k - 1, :] = perm[-batch_size:]
    a = a.astype(int)
    return a, k


# 将每一个训练数据转成统一的格式： 统一以start_char开头， 将不在词典中的词语换成oov_char， 最后使用padding_value填充至固定长度
def pre_process(x, max_L, max_words, start_char, oov_char, padding_char):
    x = [start_char] + x
    L = len(x)
    if L > max_L:
        x = x[:max_L]
    else:
        x = x + [padding_char] * (max_L - L)
    x = [oov_char if w >= max_words else w for w in x]
    return x


# 将一个数据的词序列表转换为对应的词向量矩阵
def idx_vec(x, max_L, embedding_dim, w2v):
    x = pre_process(x, max_L, max_words, start_char, oov_char, padding_char)
    a = np.zeros([max_L, embedding_dim])
    for i in range(len(x)):
        a[i, :] = w2v[int(x[i]), :]
    return a


# 将一批数据的词序列表转换为对应的词向量矩阵
def batch_process(x, max_L, embedding_dim):
    batch_size = len(x)
    a = np.zeros([batch_size, max_L, embedding_dim])
    for i in range(batch_size):
        a[i, :, :] = idx_vec(x[i], max_L, embedding_dim, w2v)
    return a


# 构建LSTM cell
def LSTM(x, weights, biases):
    # 利用unstack函数，沿着max_L指标，每次得到单个400维输入向量x
    x = tf.unstack(x, max_L, 1)
    # 这里直接调用tf.contrib.RNN中的高级API，构建一个标准的LSTM
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 得到LSTM的输出以及此时的LSTMCell细胞状态
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # outputs[-1]是最后一个LSTMCell的输出，也就是整个RNN的最终输出，即我们需要的结果
    return tf.matmul(outputs[-1], weights['LSTM_out']) + biases['LSTM_out']


# 画出混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 加载词向量矩阵
w2v_all = np.load("../data/embedding.npy")
w2v = get_new_w2v(w2v_all, max_words, embedding_dim)
print(np.shape(w2v))
print("load w2v matrix sucessfully!")

with tf.name_scope("inputs"):
    x = tf.placeholder("float", [None, max_L, embedding_dim], name="input")
    y = tf.placeholder("float", [None, n_classes], name="output")


# 定义模型的权重矩阵
with tf.name_scope("weights"):
    weights = {
        # 随机初始化一个n_hidden * n_classes维的权重矩阵
        'LSTM_out': tf.Variable(tf.random_normal([n_hidden, 100]), name="LSTM_out_weight"),
        'fc_out': tf.Variable(tf.random_normal([100, n_classes]), name="FC_out_weight")}
    tf.summary.histogram("LSTM_out_weight", weights['LSTM_out'])
    tf.summary.histogram("FC_out_weight", weights['fc_out'])


# 定义模型的偏置向量
with tf.name_scope("biases"):
    biases = {
        # 随机初始化一个n_classes维的偏置向量
        'LSTM_out': tf.Variable(tf.random_normal([100]), name="LSTM_out_bias"),
        'fc_out': tf.Variable(tf.random_normal([n_classes]), name="FC_out_bias")}
    tf.summary.histogram("LSTM_out_bias", biases['LSTM_out'])
    tf.summary.histogram("FC_out_bias", biases['fc_out'])


# 在LSTM最后一层的输出后连接一个维度为n_class的全连接层
with tf.name_scope("LSTM"):
    LSTM_output = LSTM(x, weights, biases)
    tf.summary.histogram('LSTM_output', LSTM_output)
fc_input = tf.nn.relu(tf.nn.dropout(LSTM_output, keep_prob))
pred = tf.matmul(fc_input, weights['fc_out']) + biases['fc_out']


# 定义损失函数和优化器
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('loss', cost)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# 评估模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 预测正确为1. 预测错误为0
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 正确率
    tf.summary.scalar('accuracy', accuracy)
data_label = tf.argmax(y, 1)
pred_label = tf.argmax(pred, 1)


# 初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


# 如果train=1，则进行训练
if train:
    print("start training!")
    with tf.Session() as sess:
        # 定义summary日志的保存路径
        merge_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../log', sess.graph)
        sess.run(init)
        step = 1
        while step < max_epoch:
            index, k = next_batch_index(len(data_train), batch_size)
            for i in range(k):
                # 数据处理
                batch_x = batch_process(x_train[index[i, :]], max_L, embedding_dim)
                batch_y = y_train[index[i, :]]
                # 模型训练
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                batch_x = batch_process(x_train, max_L, embedding_dim)
                batch_y = y_train
                # 计算当前的准确率
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # 计算当前的损失函数
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(step) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
            # 将每一步的数据变化记录在本地，后续用于tensorboard的可视化
            result = sess.run(merge_op, feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(result, step)
            step += 1
        saver_path = saver.save(sess, "../Saved_model/model.ckpt")
        print("Optimization Finished!")


# 如果train=0，则进行测试
if train == 0:
    with tf.Session() as sess:
        saver.restore(sess, "../Saved_model/model.ckpt")
        # 训练集上的表现
        batch_x = batch_process(x_train, max_L, embedding_dim)
        batch_y = y_train
        print("Training Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        cnf_matrix = confusion_matrix(sess.run(data_label, feed_dict={y: batch_y}),
                                      sess.run(pred_label, feed_dict={x: batch_x}))
        # 画出未归一化的混淆矩阵
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        # 画出归一化的混淆矩阵
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        plt.show()

        # 测试集上的表现
        batch_x = batch_process(x_test, max_L, embedding_dim)
        batch_y = y_test
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        cnf_matrix = confusion_matrix(sess.run(data_label, feed_dict={y: batch_y}),
                                      sess.run(pred_label, feed_dict={x: batch_x}))
        # 画出未归一化的混淆矩阵
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        # 画出归一化的混淆矩阵
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        plt.show()