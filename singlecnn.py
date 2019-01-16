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
csv_train_x=csv.reader(open('F:/audiorecognition/data/train.csv',encoding='utf-8'))
csv_train_y=csv.reader(open('F:/audiorecognition/data/train_label.csv',encoding='utf-8'))
csv_text_x=csv.reader(open('F:/audiorecognition/data/test.csv',encoding='utf-8'))
csv_text_y=csv.reader(open('F:/audiorecognition/data/test_label.csv',encoding='utf-8'))

#get all the data with form list
rows_train_x = [row for row in csv_train_x]
rows_train_y = [row for row in csv_train_y]
rows_text_x = [row for row in csv_text_x]
rows_text_y = [row for row in csv_text_y]
#change the data form from string to float
row_count = len(rows_text_x)
print(len(rows_train_x),len(rows_train_y),len(rows_text_x),len(rows_text_y),'\n')
#text_xs = [[None] for j in range(row_count)]
text_xs = np.array(rows_text_x)
text_ys = np.array(rows_text_y)


batch_size = 20

#caculate the number of batch
l=len(rows_train_x)
n_batch = l // batch_size

# def weight_variable(name,shape):
#     initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
#     return tf.get_variable(name,shape=shape,initializer=initial)

# def bias_variable(name,shape):
#     initial = tf.constant(0.0001, shape=shape)
#     return tf.get_variable(name,shape=shape,initializer=initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_new(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 1000, 1],strides=[1, 1, 1000, 1], padding='SAME')

#set input placeholder
x = tf.placeholder(tf.float32,[None,11962])
y = tf.placeholder(tf.float32,[None,2])

x_audio = tf.reshape(x, [-1,1,11962,1])

#build network
# W_conv1 = weight_variable("W_conv1",[1, 10, 1, 16])  # fiter with size [1,10] for one varialbe time series
# b_conv1 = bias_variable("b_conv1",[16])
W_conv1 = tf.get_variable("W_conv1", shape=[1, 10, 1, 16], initializer = tf.truncated_normal_initializer(stddev=0.1))
b_conv1 = tf.get_variable("b_conv1",shape=[16],initializer = tf.constant_initializer(0.0001))
h_conv1 = tf.nn.relu(conv2d(x_audio, W_conv1) + b_conv1)

h_pool1 = max_pool_new(h_conv1)

flatten = tf.reshape(h_pool1, [-1,1*12*16])

# W_fc1 = weight_variable([1*12*16,40])
# b_fc1 = bias_variable([40])
W_fc1 = tf.get_variable("W_fc1", shape=[1*12*16,40], initializer = tf.truncated_normal_initializer(stddev=0.1))
b_fc1 = tf.get_variable("b_fc1",shape=[40],initializer = tf.constant_initializer(0.0001))
h_fc1 = tf.nn.relu(tf.matmul(flatten,W_fc1) + b_fc1)

# W_fc2 = weight_variable([40,10])
# b_fc2 = bias_variable([10])
W_fc2 = tf.get_variable("W_fc2", shape=[40,10], initializer = tf.truncated_normal_initializer(stddev=0.1))
b_fc2 = tf.get_variable("b_fc2",shape=[10],initializer = tf.constant_initializer(0.0001))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

# W_fc3 = weight_variable([10,2])
# b_fc3 = bias_variable([2])
W_fc3 = tf.get_variable("W_fc3", shape=[10,2], initializer = tf.truncated_normal_initializer(stddev=0.1))
b_fc3 = tf.get_variable("b_fc3",shape=[2],initializer = tf.constant_initializer(0.0001))
prediction = tf.nn.softmax(tf.matmul(h_fc2,W_fc3) + b_fc3)


#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))

#optimizer
train_step = tf.train.AdamOptimizer(0.002).minimize(loss)

#initialize
init = tf.global_variables_initializer()

#compute accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
#train
with tf.Session() as sess:
    # sess.run(init)

    print ('Starting: %s' % datetime.datetime.now())
    # Restore variables from disk.
    saver.restore(sess, "./model/model.ckpt")
    print("Model restored.")
    batch_xs = [[] for j in range(batch_size)]
    batch_ys = [[] for j in range(batch_size)]
    for epoch in range(1):
        for batch in range(n_batch):
            #put in data
            batch_xs = np.array(rows_train_x[batch*batch_size:(batch+1)*batch_size])
            batch_ys = np.array(rows_train_y[batch*batch_size:(batch+1)*batch_size])

            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            

        acc = sess.run(accuracy,feed_dict={x:text_xs,y:text_ys})
        print('------------Iter: ' + str(epoch + 1) + '  Testing Accuracy: ' + str(acc) + '\n')
    # Save the variables to disk.
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print ('Ending: %s' % datetime.datetime.now())