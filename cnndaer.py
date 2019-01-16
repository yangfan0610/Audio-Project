import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt

#get all the data with form list
csv_train_x=csv.reader(open('F:/audioextract/data/trianinput.csv',encoding='utf-8'))
csv_train_y=csv.reader(open('F:/audioextract/data/trianoutput.csv',encoding='utf-8'))

#get all the data with form list
rows_train_x = np.array([row for row in csv_train_x])
rows_train_y = np.array([row for row in csv_train_y])

#set batch size
batch_size = 100

#caculate the number of batch
l=len(rows_train_x)
n_batch = l // batch_size
print(l,n_batch,'\n')
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0001, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_new(x,k,s):
    return tf.nn.max_pool(x, k, s, padding='SAME')

#set input placeholder
x = tf.placeholder(tf.float32,[None,30*325])
y = tf.placeholder(tf.float32,[None,30*325])

x_audio = tf.reshape(x, [-1,30,325,1])
y_audio = tf.reshape(y, [-1,30,325,1])
#build network
W_conv1 = weight_variable([3, 3, 1, 12])  
b_conv1 = bias_variable([12])
h_conv1 = tf.nn.relu(conv2d(x_audio, W_conv1) + b_conv1)
h_pool1 = max_pool_new(h_conv1,[1, 3, 5, 1],[1, 3, 5, 1])

W_conv2 = weight_variable([3, 3, 12, 20])  
b_conv2 = bias_variable([20])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_new(h_conv2,[1, 2, 5, 1],[1, 2, 5, 1])

W_conv3 = weight_variable([3, 3, 20, 30])  
b_conv3 = bias_variable([30])


W_conv1_3 = weight_variable([1, 1, 12, 30])  
b_conv1_3 = bias_variable([30])
h_pool1_3 = max_pool_new(h_pool1,[1, 2, 2, 1],[1, 2, 2, 1])
h_conv1_3 = tf.nn.relu(conv2d(h_pool1_3, W_conv1_3) + b_conv1_3)


h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_conv3_1_3 = h_conv1_3 + h_conv3

W_conv4 = weight_variable([3, 3, 30, 40])  
b_conv4 = bias_variable([40])
h_conv4 = tf.nn.relu(conv2d(h_conv3_1_3, W_conv4) + b_conv4)

W_conv5 = weight_variable([3, 3, 40, 30])  
b_conv5 = bias_variable([30])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([3, 3, 30, 20])  
b_conv6 = bias_variable([20])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
up_sampling1 = tf.image.resize_nearest_neighbor(h_conv6, (2*5,2*50))

W_conv7 = weight_variable([3, 3, 20, 12])  
b_conv7 = bias_variable([12])
h_conv7 = tf.nn.relu(conv2d(up_sampling1, W_conv7) + b_conv7)
up_sampling2 = tf.image.resize_nearest_neighbor(h_conv7, (3*10,2*100))

W_conv8 = weight_variable([3, 3, 12, 1]) 
b_conv8 = bias_variable([1])
logits_ = conv2d(up_sampling2, W_conv8) + b_conv8
prediction = tf.nn.sigmoid(logits_)

# W_conv9 = weight_variable([3, 3, 1, 1])  
# b_conv9 = bias_variable([1])
# h_conv9 = tf.nn.sigmoid(conv2d(h_conv8, W_conv9) + b_conv9)

#loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_audio,logits=logits_))

#optimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#initialize
init = tf.global_variables_initializer()


#train
with tf.Session() as sess:
    sess.run(init)

    print ('Starting: %s' % datetime.datetime.now())
    batch_xs = [[] for j in range(batch_size)]
    batch_ys = [[] for j in range(batch_size)]
    for epoch in range(20):
        for batch in range(n_batch):
            #put in data
            batch_xs = np.array(rows_train_x[batch*batch_size:(batch+1)*batch_size])
            batch_ys = np.array(rows_train_y[batch*batch_size:(batch+1)*batch_size])
         
            
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        if (epoch+1) % 20 == 0:
            loss_batch = [[] for j in range(32)]
            for j in range(32):
                loss_batch[j] = sess.run(loss,feed_dict={x:rows_train_x[j*200:(j+1)*200],y:rows_train_y[j*200:(j+1)*200]})
            loss_all = sum(loss_batch)/32
            # loss_all = sess.run(loss,feed_dict={x:rows_train_x,y:rows_train_y})
            print('------------Iter: ' + str(epoch + 1) + '  Testing Loss: ' + str(loss_all) + '\n')


    #plot picture
    x_audio,y_audio,prediction = sess.run([x_audio,y_audio,prediction],feed_dict={x:rows_train_x[:2],y:rows_train_y[:2]})

    fig,axes = plt.subplots(nrows = 3,ncols = 2,figsize = (20,8))
    for image,row in zip([x_audio,y_audio,prediction],axes):
        for img,ax in zip(image,row):
            ax.imshow(img.reshape((30,200)),cmap = 'Greys_r')
            ax.set_aspect('auto')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.tight_layout() 
    plt.show() 


    print ('Ending: %s' % datetime.datetime.now())