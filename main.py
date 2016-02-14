import models
import cPickle
import numpy as np
import tensorflow as tf

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def one_hot_vec(label):
    vec = np.zeros(10)
    vec[label] = 1
    return vec

def load_data():
    x_all = []
    y_all = []
    for i in range (5):
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))
        x_ = d['data']
        y_ = d['labels']
        x_all.append(x_)
        y_all.append(y_)

    d = unpickle('cifar-10-batches-py/test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    y = map(one_hot_vec, y)
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return (X_train, Y_train, X_test, Y_test)

batch_size = 100
X_train, Y_train, X_test, Y_test = load_data()

X = tf.placeholder("float", [batch_size, 32, 32, 3])
Y = tf.placeholder("float", [batch_size, 10])
net = models.resnet20(X)

cross_entropy = -tf.reduce_sum(Y*tf.log(net))
opt = tf.train.MomentumOptimizer(0.01, 0.9)
train_op = opt.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(".")
if checkpoint:
    print "restoring from checkpoint", checkpoint
    saver.restore(sess, checkpoint)
else:
    print "couldn't find checkpoint to restore from."

for i in range (0, 50000, batch_size):
    feed_dict={
        X: X_train[i:i + batch_size], 
        Y: Y_train[i:i + batch_size]}
    sess.run([train_op], feed_dict=feed_dict)
    if i % 1000 == 0:
        saver.save(sess, 'progress', global_step=i+50000)

for i in range (0, 10000, batch_size):
    acc =  sess.run([accuracy],feed_dict={
        X: X_test[i:i+batch_size],
        Y: Y_test[i:i+batch_size]
    })
    print acc


sess.close()