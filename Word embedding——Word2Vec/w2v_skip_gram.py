#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np 
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

filename = 'text8.zip'

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data      # read the data into a list of strings

words = read_data(filename)


# step1 剔除高频停用词减少模型噪音，并加速训练
# 感觉可以在分词时就进行停用词的处理
def remove_fre_stop_word(words):
    t = 1e-5   # t 值
    threshold = 0.8   # 剔除概率的阈值
    # 统计单词频率
    int_word_counts = collections.Counter(words)
    total_count = len(words)
    word_freqs = {w:c/total_count for w,c in int_word_counts.items()}
    # 计算被删除概率
    prob_drop = {w:1-np.sqrt(t/f) for w,f in word_freqs.items()}
    # 对单词进行采样
    train_words = [w for w in words if prob_drop[w] < threshold]

    return train_words

words = remove_fre_stop_word(words)



# step2 建立词典，并把低频次用UNK代替
vocabulary_size = 150000  # 只留150000个不重复的词语，其余归为UNK
def build_dataset(words):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size -1))
    dictionary = dict()
    # 按照词频，给每个词分配一个编号：（‘UNK’：0），（‘的’：1） 。。。
    for word,_ in count:
        dictionary[word] = len(dictionary)    # 词对编号

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            inex = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # 编号对词
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary


data,count,dictionary,reverse_dictionary = build_dataset(words)

del words   # reduce memory

data_index = 0

# step3:Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window + 1   # [skip_window  target  skoip_window]
    buffer = collections.deque(maxlen=span)    # 类似于list
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index +1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])                        # buffer队列，先进先出，永远保持5个
        data_index = (data_index + 1) % len(data)
    data_index -= 1
    return batch,labels



# step 4:Build and train a skip_gram model.
# hyperparameters
batch_size = 128
embedding_size = 300
skip_window = 2
num_skips = 4

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window,valid_size)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    embed = tf.nn.embedding_lookup(embeddings,train_inputs)

    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(nce_weights,nce_biases,train_labels,
                        embed,num_sampled,vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

num_steps = 100001
final_embeddings = []

with tf.Session(graph=graph) as session:
    # we must initialize all variables before using them
    tf.initialize_all_variables().run()
    print('initialized.')
    
    # loop through all training steps and keep track of loss
    average_loss = 0
  
    for step in xrange(num_steps):
        # generate a minibatch of training data
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # we perform a single update step by evaluating the optimizer operation (including it
        # in the list of returned values of session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val


        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # the average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        
        # computing cosine similarity (expensive!)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                # get a single validation sample
                valid_word = reverse_dictionary[valid_examples[i]]
                # number of nearest neighbors
                top_k = 8
                # computing nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary.get(nearest[k],None)
                    #close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        
    final_embeddings = normalized_embeddings.eval()
    print("*"*10+"final_embeddings:"+"*"*10+"\n",final_embeddings)
    fp=open('vector_skip_gram.txt','w',encoding='utf8')
    for k,v in reverse_dictionary.items():
        t=tuple(final_embeddings[k])

        s=''
        for i in t:
            i=str(i)
            s+=i+" "
            
        fp.write(v+" "+s+"\n")

    fp.close()



# Step 6: Visualize the embeddings.
import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, plot_labels, filename='tsne_skip_gram.png'):
    assert low_dim_embs.shape[0] >= len(plot_labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(plot_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(u'{}'.format(label),
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)




try:
    from sklearn.manifold import TSNE
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文字符
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示正负号

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    plot_labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, plot_labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
