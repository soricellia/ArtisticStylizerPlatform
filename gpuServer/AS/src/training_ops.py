import tensorflow as tf
import module_utils

class training_ops:
    def __init__(self, FLAGS):
        self.mod_utils = module_utils.module_utils(FLAGS)  # initialize project constants
        self.init_queue()
    # end

    def init_queue(self):
        # Setup data loading queue
        self.queue_input_content = tf.placeholder(tf.float32, shape=self.mod_utils.batch_shape)
        self.queue_input_style = tf.placeholder(tf.float32, shape=self.mod_utils.batch_shape)
        self.queue = tf.FIFOQueue(capacity=500,
                                  dtypes=[tf.float32, tf.float32],
                                  shapes=[self.mod_utils.batch_shape[1:], self.mod_utils.batch_shape[1:]])
        self.enqueue_op = self.queue.enqueue_many([self.queue_input_content, self.queue_input_style])
        self.dequeue_op = self.queue.dequeue()
        self.content_batch_op, self.style_batch_op = tf.train.batch(self.dequeue_op,
                                                                    batch_size=self.mod_utils.batch_size,
                                                                    capacity=500)
    # end

    def enqueue(self, sess):
        content_images = self.mod_utils.get_batch(dir=self.mod_utils.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR)
        style_images = self.mod_utils.get_batch(dir=self.mod_utils.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR)
        while True:
            content_batch = next(content_images)
            style_batch = next(style_images)

            sess.run(self.enqueue_op, feed_dict={self.queue_input_content: content_batch,
                                            self.queue_input_style:   style_batch})
    # end 
# end 
