import tensorflow as tf
import numpy as np
import threading
import module_utils
import os

class training_ops:
    def __init__(self, FLAGS):

        self.mod_utils = module_utils.module_utils(FLAGS) # initialize project contants
        self.all_training_content_images = self.mod_utils.load_data(dir=self.mod_utils.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR)
        self.all_training_style_images = self.mod_utils.load_data(dir=self.mod_utils.IMG_TRAIN_WIKIART_STYLE_DIR)

        for i in range(self.mod_utils.num_epochs):
            batch = self.mod_utils.get_batch_iter(data=self.all_training_content_images,
                                          batch_size=self.mod_utils.batch_size,
                                          num_epochs=self.mod_utils.num_epochs)
            print(batch)
                                          



        with tf.Graph().as_default():
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)
            self.sess = tf.Session(config=session_config)
            with self.sess.as_default():
                # global step 
                self.global_step = tf.Variable(0, name="global_step", trainable=False)

                # checkpoint model
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.mod_utils.num_chkpt)
    # end 

    def train_step(self, x_batch, y_batch):
        print(x_batch)
    # end 
# end 
