"""
taken from https://github.com/machrisaa/tensorflow-vgg.git
"""

import numpy as np
import tensorflow as tf

import vgg16
import utils

img1 = utils.load_image("../img/content/hancockCenter.jpg")
img2 = utils.load_image("../img/content/pabloRivas.jpg")
img3 = utils.load_image("../img/content/route9tunnel.jpg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch3 = img3.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2, batch3), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
# TODO update this
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [3, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)
        utils.print_prob(prob[0], './synset.txt')
        utils.print_prob(prob[1], './synset.txt')
