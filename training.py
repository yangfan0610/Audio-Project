'''
the aim is to recognize the audio
version  4.0
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import datetime
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
csv_train_x=csv.reader(open('F:/audiorecognition/data/dataset13.csv',encoding='utf-8'))
csv_train_y=csv.reader(open('F:/audiorecognition/data/label13.csv',encoding='utf-8'))

#get all the data with form list
rows_train_x = np.array([row for row in csv_train_x])
rows_train_y = np.array([row for row in csv_train_y])

X_train, X_test, y_train, y_test = train_test_split(rows_train_x, rows_train_y, test_size=0.15, random_state=42)
#change the data form from string to float
print(len(X_train),len(X_test),'\n')

#set batch size
batch_size = 20

#caculate the number of batch
l=len(X_train)
n_batch = l // batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_new(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 100, 1],strides=[1, 1, 100, 1], padding='SAME')

#set input placeholder
x = tf.placeholder(tf.float32,[None,16000])
y = tf.placeholder(tf.float32,[None,2])

x_audio = tf.reshape(x, [-1,1,16000,1])

#build network
W_conv1 = weight_variable([1, 10, 1, 32])  # fiter with size [1,10] for one varialbe time series
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_audio, W_conv1) + b_conv1)

h_pool1 = max_pool_new(h_conv1)

W_conv2 = weight_variable([1, 10, 32, 16])  # fiter with size [1,10] for one varialbe time series
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 10, 1],strides=[1, 1, 10, 1], padding='SAME')

flatten = tf.reshape(h_pool2, [-1,1*16*16])


W_fc1 = weight_variable([1*16*16,40])
b_fc1 = bias_variable([40])
h_fc1 = tf.nn.relu(tf.matmul(flatten,W_fc1) + b_fc1)

W_fc2 = weight_variable([40,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

W_fc3 = weight_variable([10,2])
b_fc3 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc2,W_fc3) + b_fc3)


#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))

#optimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#initialize
init = tf.global_variables_initializer()

#compute accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#train
with tf.Session() as sess:
    sess.run(init)

    print ('Starting: %s' % datetime.datetime.now())
    for epoch in range(15):
        for batch in range(n_batch+1):
            #put in data  
            if batch == n_batch:    
                batch_xs = X_train[batch*batch_size:]
                batch_ys = y_train[batch*batch_size:]
            else:
                batch_xs = X_train[batch*batch_size:(batch+1)*batch_size]
                batch_ys = y_train[batch*batch_size:(batch+1)*batch_size]
           
         
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        dataset = list(zip(X_train, y_train))
        random.shuffle(dataset)
        X_train[:], y_train[:] = zip(*dataset)
        
        acc1 = sess.run(accuracy,feed_dict={x:X_test[:250],y:y_test[:250]})
        acc2 = sess.run(accuracy,feed_dict={x:X_test[251:500],y:y_test[251:500]})
        acc = (acc1 + acc2)/2
        print('------------Iter: ' + str(epoch + 1) + '  Testing Accuracy: ' + str(acc) + '\n')

    print ('Ending: %s' % datetime.datetime.now())