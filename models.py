import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block

# ResNet architectures used for CIFAR-10
def resnet20(inpt):

    with tf.variable_scope("conv1"):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)

    with tf.variable_scope("conv2"):
        conv2_1 = residual_block(conv1, 16, False)
        conv2 = residual_block(conv2_1, 16, False)

    with tf.variable_scope("conv3"):
        conv3_1 = residual_block(conv2, 32, False)
        conv3 = residual_block(conv3_1, 32, False)

    with tf.variable_scope("conv4"):
        conv4_1 = residual_block(conv3, 64, False)
        conv4 = residual_block(conv4_1, 64, False)

    with tf.variable_scope("out"):
        global_pool = tf.reduce_mean(conv4, [1, 2])
        out = softmax_layer(global_pool, [64, 10])

    return out

def resnet32(inpt):
    pass

def resnet44(inpt):
    pass

def resnet56(inpt):
    pass

def resnet110(inpt):
    pass

