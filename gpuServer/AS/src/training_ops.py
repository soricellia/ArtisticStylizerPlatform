import tensorflow as tf
import module_utils
import os

class training_ops:
    def __init__(self, FLAGS):

        with tf.Graph().as_default():
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)
            sess = tf.Session(config=session_config)
            with sess.as_default():

                self.mod_utils = module_utils.module_utils(FLAGS)


                coordinator = tf.train.Coordinator() # use coordinator to manage loading image files
                threads = tf.train.start_queue_runners(coord=coordinator)

                with tf.name_scope("load_data"):
                    jpg_files = [os.path.join(self.mod_utils.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR, file) 
                            for file in os.listdir(self.mod_utils.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR)
                            if file.endswith(".jpg")]
                    filename_queue = tf.train.string_input_producer(jpg_files)
                    image_reader = tf.WholeFileReader() # intialize file reader
                    filename, image_file = image_reader.read(filename_queue) # return filename and pixel values as tensor
                    image = tf.image.decode_jpeg(image_file, channels=3)
                    resized_image = tf.image.resize_images(images=image, 
                                                size=[self.mod_utils.img_height_dim,
                                                      self.mod_utils.img_width_dim])
                    resized_image_squeeze = tf.squeeze(resized_image)

                    with tf.name_scope("get_batches"):
                        for i in range(self.mod_utils.num_epochs):

                            for i in range((round(len(jpg_files)/self.mod_utils.batch_size))):
                                batch_results, _ = tf.train.shuffle_batch([resized_image_squeeze, filename], 
                                                              batch_size=self.mod_utils.batch_size,
                                                              num_threads=4,
                                                              capacity=10000,
                                                              min_after_dequeue=5000)
                                print(sess.run([batch_results]))


            # kill coordinator and join threads
            coordinator.request_stop() 
            coordinator.join(threads)
# end 
