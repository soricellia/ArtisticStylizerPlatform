"""
original author (https://arxiv.org/pdf/1703.06868.pdf) implementation
uses normalized vgg19 found here  https://s3.amazonaws.com/xunhuang-public/adain/vgg_normalised.t7
"""
import os
import tensorflow as tf

import numpy as np
import time
import inspect

import torchfile

def vgg19(input, t7_file, scope, type):
    '''
    Loads a Torch network from a saved .t7 file into Tensorflow.

    :param input Input to Torch network
    :param t7 Path to t7 file to use
    '''
    with tf.variable_scope("{}_vgg19".format(scope)):
        print('enter vgg19')
        print(input.shape)
        layers = []
        print_layers = []  # [0, 30]
        t7 = torchfile.load(t7_file, force_8bytes_long=False)

        for idx, module in enumerate(t7.modules):

            if idx in print_layers:
                print('layers')
                print(module)

            if module._typename == b'nn.SpatialReflectionPadding':
                left = module.pad_l
                right = module.pad_r
                top = module.pad_t
                bottom = module.pad_b
                input = tf.pad(input, [[0, 0], [top, bottom], [left, right], [0, 0]], 'REFLECT')
                layers.append(input)
            elif module._typename == b'nn.SpatialConvolution':
                weight = module.weight.transpose([2, 3, 1, 0])
                bias = module.bias
                strides = [1, module.dH, module.dW, 1]  # Assumes 'NHWC'
                input = tf.nn.conv2d(input, weight, strides, padding='VALID')
                input = tf.nn.bias_add(input, bias)
                layers.append(input)
            elif module._typename == b'nn.ReLU':
                input = tf.nn.relu(input)
                layers.append(input)
            elif module._typename == b'nn.SpatialUpSamplingNearest':
                d = tf.shape(input)
                size = [d[1] * module.scale_factor, d[2] * module.scale_factor]
                input = tf.image.resize_nearest_neighbor(input, size)
                layers.append(input)
            elif module._typename == b'nn.SpatialMaxPooling':
                input = tf.nn.max_pool(input, ksize=[1, module.kH, module.kW, 1], strides=[1, module.dH, module.dW, 1],
                                     padding='VALID', name=str(module.name, 'utf-8'))
                layers.append(input)
            else:
                raise NotImplementedError(module._typename)
        output = input
        if type == "encoder":
            return layers
        else:
            return output
# end

"""
original tensorflow implementation of vgg19 found here https://github.com/machrisaa/tensorflow-vgg.git
"""

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

