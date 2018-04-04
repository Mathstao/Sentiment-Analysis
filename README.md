## 					  					                      利用Word2Vec以及LSTM模型对商品评论进行情感分析

##### 								   						     	                                                                                                                                                                                                                       肖舒月     14336215     应用统计

#### 1. 实验环境

- Window 10 x64
- Python 3.6.3
- gensim 3.2.0
- jieba 0.39
- tensorflow 1.4.0 & tensorboard 0.4.0
- pandas 0.20.32




####2. 研究目的

随着互联网的发展，人们越来越倾向于在网上表达自己的情绪与观点。在舆情分析、产品反馈等多个应用场景下都需要我们知道用户的情感，这就是自然语言处理中的“情感分析”任务，同时它也是人机交互中的一个重要课题。

其中，情感分析问题主要有两个难点：

- 数据量大，直接用借助人力去分析的话十分耗时耗力。 
- 机器难以理解人类高级语言的含义，导致规则方法或者情感词频统计法等传统方法的准确率不高。

针对以上两种情况，本文尝试将深度学习运用在情感分析任务上，利用收集到的足够多的数据，构建长短期记忆网络（LSTM）模型，得到一个自动化的且准确率较高的情感分析判别模型。



#### 3. 研究流程

1. 收集网上开源的中文文本情感分析数据集
2. 调研近几年深度学习网络在情感分析上的应用，并对比各模型的效果，选定需要实现的模型
3. 确定实现模型的深度学习框架
4. 实现模型并分析实验结果




#### 4. 基本方法描述

1. 词向量模型一种基于神经网络的语言模型，它可以把词语建模成指定维度的向量，并且语义相近的词语在向量空间中的余弦距离会十分接近。借助词向量我们可以将机器难以理解的文本转换成易于理解的数值向量特征。
2. LSTM模型是循环神经网络（RNN）的一个变形，适合于处理和预测时间序列中间隔和延迟相对较长的重要事件，也正是由于这个特点，LSTM在这各种关于序列预测的问题上往往会取得不错的效果。本文中利用LSTM的最后一层的输出，将其理解为模型对于句子在情感方面抽取的特征，并将这些特征再与2个神经元全连接，最后利用Softmax方法得到最终预测结果。




#### 5. 数据说明

本次实验训练数据均来源于网上，内容为用户对于商品的评论，其中正向评论有10679条、负向评论有10428条，数据分布均衡。另外，中文停用词典stopword_ch.txt也来源于网上，包括全部常见标点符号以及无意义的词语。

训练数据样式如下：

- (postive) 我送了一本给我阿姨，她看了后大很好，大有启发，看来我多了一样送长辈的东西了，所以后来我又买了四本，准备送其他长辈，看来送朋友也很不错喔，正好现在流行如何养生呢。
- (positve) 电热水器非常好，性价比高，顾客买的放心。
- (negative) 美的售后太垃圾，其他售后都是两小时回电话，美的是24小时，结果超过市区不到五公里问我收五十，服务很差。再不买美的的东西了。卖家服务还不错。
- (negative) 1.不支持MP3铃声。2.开启程序时速度较慢。3.2M的内存真让人有些伤感。




####6. 实验思路

1. 对全部评论进行整合、分词、去停用词，得到语料库。
2. 利用语料库进行Word2Vec的训练，得到词向量模型，同时建立相应的word-idx字典。
3. 将整合后的评论重新排列(shuffle)，然后按照8:2的比例划分训练集和测试集，并且根据word-idx将句子转成index列表，得到训练集train.npy和测试集test.npy。
4. 基于tensorflow建立LSTM模型，训练得到最终的结果并可视化。




#### 7. 实验结果分析

##### （1）模型网络结构可视化：

![model graph](C:\Users\Citao\code\Sentiment Analysis\model graph.png)



##### （2）迭代50次后，模型趋向稳定，最终在训练集和训练集上取得的准确率如下：

- Training Accuracy: 0.845899
- Testing Accuracy: 0.819517

在未经过仔细调整模型参数的情况下，准确率可达到80%以上，说明我们的模型在情感分析问题上的应用是可行的。后续如果再增加训练集的数量以及做好调参的工作，准确率应该可以达到90%以上。



##### （3）未归一化的混淆矩阵分别如下：

![confusion matrix](C:\Users\Citao\code\Sentiment Analysis\confusion matrix.png)

结果显示我们的模型对于正向情感的识别能力强于负向情感。这也是可以解释的，因为人们在表达正向情感时，往往会比较直接，机器容易识别。而有一些人在表达负向情感时，喜欢用婉转或者嘲讽的方式，这时候机器就会难以判别。



##### （4）准确率和损失函数变化

![loss](C:\Users\Citao\code\Sentiment Analysis\loss.png)

在实验过程中，发现了一个很有趣的现象：模型初期训练提升十分有限，准确率仅有60%波动。但在第15次迭代左右时，准确率和损失函数均有跳跃式的提升，猜想是此时模型跳出了局部最优解，成功跳到了全局最优解附近。



##### （5）模型各参数变化

![histagram](C:\Users\Citao\code\Sentiment Analysis\histagram.png)



####8. 核心代码

#####（1）数据预处理

```python
# -*- coding: utf-8 -*-
import sys
import jieba
import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# 加载停用词典
stopwords = [i.strip("\n") for i in open("stopwords_ch.txt","r", encoding="utf-8").readlines()]

# 分词
def segment(x):
    return [w.strip() for w in jieba.cut(x) if w not in stopwords]
  
# 读取pos.xls和neg.xls原始评论数据，以字典的形式返回分词后的列表
def load_data(ori_data_path):
    pos_data = xlrd.open_workbook(ori_data_path + "pos.xls")
    pos_table = pos_data.sheets()[0]
    pos_sentences = []
    for i in range(pos_table.nrows):
        x = pos_table.cell(i,0).value
        pos_sentences.append(segment(x))
    neg_data = xlrd.open_workbook(ori_data_path + "neg.xls")
    neg_table = neg_data.sheets()[0]
    neg_sentences = []
    for i in range(neg_table.nrows):
        x = neg_table.cell(i,0).value
        neg_sentences.append(segment(x))
    return {"pos":pos_sentences, "neg":neg_sentences} 

# 得到用于训练词向量的语料库并保存为corpus.txt
def get_corpus(corpus_file, sentences):
    fout = open(corpust_file, "w", encoding="utf-8")
    for i in ["pos","neg"]:
        temp = sentences[i]
        for j in temp:
            fout.write(" ".join(j) + "\n")
    fout.close()
    return None
  
# 根据词频数降序排序，得到{word:index}和{index:word}两个字典，方便后续的数据处理
def get_word_with_idx_dicts(sentences):
    lists = []
    for i in sentences["pos"]+sentences["neg"]:
        lists.extend(i)
    dicts = Counter(lists)
    sort_dict = sorted(dicts.items(),key=lambda x:x[1], reverse=1)
    word_list = [item[0] for item in sort_dict]
    word_idx = {word_list[i]:i+1 for i in range(len(word_list))}
    idx_word = {v:k for k,v in word_idx.items()}
    return word_idx, idx_word

# 将一个句子分词列表根据word_idx转换为词序号列表
def word_list_to_idx(words, word_idx):
  	word_list = [item[1] for item in word_idx]
    return [word_idx[word] if word in word_list else 0 for word in words]

# 训练词向量模型，size是词向量的维度，window是模型的窗口大小，词频数低于min_count的单词不会被保留  
def train_w2v_model(corpus_file, model_file, size, window, min_count):
    sen = [i.strip("\n").split(" ") for i in open(corpus_file,"r", encoding="utf-8")]
    model = Word2Vec(sentences=sen, size=size, window=window, min_count=min_count, workers=4,)
    model.save(model_file)
    return model

 # 根据word_idx和词向量模型得到词向量矩阵并保存在本地，方便后续LSTM训练时调用 
 def save_embedd_matrix(embedd_matrix_file, word_idx, model):
    embedding = np.zeros([len(word_idx)+1, 200])
  	embedding[0,:] = np.random.random(200)
  	for i in range(len(idx_word)):
      	embedding[i+1, :] = model.wv[idx_word[i+1]]
	np.save(embedd_matrix_file, embedding)
	return None

# 根据指定的比例ratio划分训练集和测试集  
def save_train_test_data(split_data_path, ratio, sentences, word_idx):
    word_list = [item[1] for item in word_idx]
    all_data_with_label = []
    label_idx = {"pos":[1,0], "neg":[0,1]}
    for label in label_idx.keys():
        for word_list in sentences[label]:
            all_data_with_label.append([word_list_to_idx(word_list, word_idx), label_idx[label]])
    all_data_with_label = np.array(all_data_with_label)
    np.random.shuffle(all_data_with_label)
    L = len(all_data_with_label)
    train_data = all_data_with_label[:int(ratio*L)]
    test_data = all_data_with_label[int(ratio*L):]
    np.save(split_data_path+"train_data.npy", train_data)
    np.save(split_data_path+"test_data.npy", test_data)
    return None

# 主函数部分
if __name__ == '__main__':
	curr_path = sys.path[0]
    sentences = load_data(curr_path)
    get_corpus(curr_path+"corpus.txt", sentences)
    word_idx, idx_word = get_word_with_idx_dicts(sentences)
    model = train_w2v_model(curr_path+"corpus.txt", curr_path+"model", size=200, window=5, min_count=1)
    save_embedd_matrix(curr_path+"embedding.npy", word_idx, model)
    save_train_test_data(curr_path, 0.8, sentences, word_idx)
```



##### （2）构建LSTM模型

```python
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

curr_path = sys.path[0]

# 读取数据
data_train = np.load(curr_path + 'train_data.npy')
data_test = np.load(curr_path + 'test_data.npy')
x_train = data_train[:,0]
temp = data_train[:,1]
y_train = np.zeros([len(temp),2])
for i in range(len(temp)):
    y_train[i,:] = temp[i]
    
x_test = data_test[:,0]
temp = data_test[:,1]
y_test = np.zeros([len(temp),2])
for i in range(len(temp)):
    y_test[i,:] = temp[i]

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))
print("load data sucessfully!")

# 模型参数
learning_rate = 0.001	 # 学习率       
train = 0			    # 训练/测试	
batch_size = 256		# 批大小
max_epoch = 50			# 最大迭代次数
display_step = 1		# 模型保存及打印间隔        
keep_prob = 0.8			# 训练时神经元的保存概率（防止过拟合）
embedding_dim = 200		# 词向量维度
max_L = 40               # 句子统一长度
max_words = 30000		# 只取词频数最高的若干个词
n_hidden = 256			# LSTM中神经元的维度
n_classes = 2			# 类别数量
class_names=["positive", "negative"]
padding_value = 0		# 句子填充的标记
start_char = 0			# 句子开始的标记
oov_char = 0			# 句子词语替换的标记

# 读取词向量矩阵
def get_new_w2v(w2v, max_words, embedding_dim):
    if np.shape(w2v)[1] != embedding_dim:
        print('The shape of w2v is not corresponding with the embedding dimension!')
        return None
    elif (np.shape(w2v)[0]+3) < max_words:
        print('Max_words is too large!')
        return None
    else:
        return w2v[:max_words,:]

# 得到在一次迭代中各批训练数据的下标
def next_batch_index(num, batch_size):
    k = math.ceil(num / batch_size)
    res = batch_size*k-num
    a = np.zeros( (k,batch_size) )
    perm = np.random.permutation(num)
    for i in range(k-1):
        a[i,:] = perm[batch_size*i:batch_size*(i+1)]
    temp = perm[-res:]
    if res == 0:
        a[k-1,:] = perm[-batch_size:]
    elif res!=0:
        while len(temp) < batch_size:
            temp = np.append(temp,np.random.randint(0, num-1))
            a[k-1,:] = perm[-batch_size:]
    a=a.astype(int)
    return a,k

# 将每一个训练数据转成统一的格式
def pre_process(x, max_L, max_words, start_char, oov_char, padding_value):
    x = [start_char] + x
    L = len(x)
    if L>max_L:
        x =  x[:max_L]
    else:
        x =  x + [padding_value] * (max_L-L)
    x = [oov_char if w >= max_words else w for w in x]
    return x
 
# 将一个数据的词序列表转换为对应的词向量矩阵
def idx_vec(x, max_L, embedding_dim, w2v):
    x = pre_process(x, max_L,  max_words, start_char, oov_char, padding_value)
    a = np.zeros([max_L, embedding_dim])
    for i in range(len(x)):
        a[i,:] = w2v[int(x[i]),:]
    return a
  
# 将一批数据的词序列表转换为对应的词向量矩阵
def batch_process(x, max_L, embedding_dim):
    batch_size = len(x)
    a = np.zeros([batch_size, max_L, embedding_dim])
    for i in range(batch_size):
        a[i,:,:] = idx_vec(x[i], max_L, embedding_dim, w2v)
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
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 加载词向量矩阵
w2v_all = np.load(curr_path + "embedding.npy")
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
fc_input =tf.nn.relu(tf.nn.dropout(LSTM_output, keep_prob))
pred = tf.matmul(fc_input, weights['fc_out']) + biases['fc_out']

# 定义损失函数和优化器
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('loss', cost)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估模型
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))     # 预测正确为1. 预测错误为0
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # 正确率
    tf.summary.scalar('accuracy', accuracy)   
data_label = tf.argmax(y,1)
pred_label = tf.argmax(pred,1)

# 初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)\

# 如果train=1，则进行训练
if train:
    print("start training!")
    with tf.Session() as sess:
        # 定义summary日志的保存路径
        merge_op = tf.summary.merge_all()    
        writer = tf.summary.FileWriter('./log', sess.graph)                    
        sess.run(init)
        step = 1
        while step < max_epoch :
            index, k = next_batch_index(len(data_train), batch_size)
            for i in range(k):
                # 数据处理
                batch_x = batch_process(x_train[index[i,:]], max_L, embedding_dim)
                batch_y = y_train[index[i,:]]
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
        saver_path = saver.save(sess, "Saved_model/model.ckpt")
        print("Optimization Finished!")

# 如果train=0，则进行测试
if train == 0:
    with tf.Session() as sess:
        saver.restore(sess, "Saved_model/model.ckpt")
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
```









