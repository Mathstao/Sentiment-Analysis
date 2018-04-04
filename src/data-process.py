# -*- coding: utf-8 -*-

import sys
import jieba
import xlrd
import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# 加载停用词典
stopwords = [i.strip("\n") for i in open("../vocab/stopwords_ch.txt", "r", encoding="utf-8").readlines()]


# 分词
def segment(x):
    return [w.strip() for w in jieba.cut(x) if w not in stopwords]


# 读取pos.xls和neg.xls原始评论数据，以字典的形式返回分词后的列表
def load_data(ori_data_path):
    pos_data = xlrd.open_workbook(ori_data_path + "pos.xls")
    pos_table = pos_data.sheets()[0]
    pos_sentences = []
    for i in range(pos_table.nrows):
        x = pos_table.cell(i, 0).value
        pos_sentences.append(segment(x))
    neg_data = xlrd.open_workbook(ori_data_path + "neg.xls")
    neg_table = neg_data.sheets()[0]
    neg_sentences = []
    for i in range(neg_table.nrows):
        x = neg_table.cell(i, 0).value
        neg_sentences.append(segment(x))
    return {"pos": pos_sentences, "neg": neg_sentences}


# 得到用于训练词向量的语料库并保存为corpus.txt
def get_corpus(corpus_file, sentences):
    f = open(corpus_file, "w", encoding="utf-8")
    for i in ["pos", "neg"]:
        temp = sentences[i]
        for j in temp:
            f.write(" ".join(j) + "\n")
    f.close()
    return None


# 根据词频数降序排序，得到{word:index}和{index:word}两个字典，方便后续的数据处理
def get_word_with_idx_dicts(sentences):
    lists = []
    for i in sentences["pos"] + sentences["neg"]:
        lists.extend(i)
    dicts = Counter(lists)
    sort_dict = sorted(dicts.items(), key=lambda x: x[1], reverse=1)
    word_list = [item[0] for item in sort_dict]
    word_idx = {word_list[i]: i + 1 for i in range(len(word_list))}
    idx_word = {v: k for k, v in word_idx.items()}
    return word_idx, idx_word


# 将一个句子分词列表根据word_idx转换为词序号列表
def word_list_to_idx(words, word_idx, word_list):
    return [word_idx[word] if word in word_list else 0 for word in words]


# 训练词向量模型，size是词向量的维度，window是模型的窗口大小，词频数低于min_count的单词不会被保留
def train_w2v_model(corpus_file, model_file, size, window, min_count):
    sen = [i.strip("\n").split(" ") for i in open(corpus_file, "r", encoding="utf-8")]
    model = Word2Vec(sentences=sen, size=size, window=window, min_count=min_count, workers=4, )
    model.save(model_file)
    return model


# 根据word_idx和词向量模型得到词向量矩阵并保存在本地，方便后续LSTM训练时调用
def save_embedd_matrix(embedd_matrix_file, word_idx, model):
    embedding = np.zeros([len(word_idx) + 1, 200])
    embedding[0, :] = np.random.random(200)
    for i in range(len(idx_word)):
        embedding[i + 1, :] = model.wv[idx_word[i + 1]]
    np.save(embedd_matrix_file, embedding)
    return None


# 根据指定的比例ratio划分训练集和测试集
def save_train_test_data(split_data_path, ratio, sentences, word_idx):
    word_list = [item[0] for item in word_idx.items()]
    all_data_with_label = []
    label_idx = {"pos":[1,0], "neg":[0,1]}
    for label in label_idx.keys():
        for words in sentences[label]:
            all_data_with_label.append([word_list_to_idx(words, word_idx, word_list), label_idx[label]])
    all_data_with_label = np.array(all_data_with_label)
    np.random.shuffle(all_data_with_label)
    L = len(all_data_with_label)
    train_data = all_data_with_label[:int(ratio*L)]
    test_data = all_data_with_label[int(ratio*L):]
    np.save(split_data_path+"train_data.npy", train_data)
    np.save(split_data_path+"test_data.npy", test_data)
    return None

import os
# 主函数部分
if __name__ == '__main__':
    curr_path = sys.path[0]
    data_path = curr_path + '/../data/'
    model_path = curr_path + '/../w2v_model/'
    sentences = load_data(data_path)
    get_corpus(data_path + "corpus.txt", sentences)
    word_idx, idx_word = get_word_with_idx_dicts(sentences)
    model = train_w2v_model(data_path + "corpus.txt", model_path + "model", size=200, window=5, min_count=1)
    save_embedd_matrix(data_path + "embedding.npy", word_idx, model)
    save_train_test_data(data_path, 0.8, sentences, word_idx)