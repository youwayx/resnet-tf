import models
import cPickle
import numpy as np
import tensorflow as tf

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    x_all = []
    y_all = []
    for i in range (5):
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))
        x = d['data']
        y = d['labels']
        x_all.append(x)
        y_all.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return (X_train, Y_train, X_test, Y_test)

X_train, Y_train, X_test, Y_test = load_data()

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

net = models.resnet20(X)

cross_entropy = -tf.reduce_sum(Y*tf.log(net))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

mini_batch_size = 100
for i in range (0, 50000, mini_batch_size):
    train_step.run(
        feed_dict={
            X: X_train[i:i + mini_batch_size], 
            Y: Y_train[i: i + mini_batch_size]})

