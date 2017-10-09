import tensorflow as tf
import numpy as np
import os

class module_paths:
    def __init__(self):
        self.SRC_DIR = os.path.abspath(os.curdir)
        self.ROOT_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.IMG_DIR = os.path.join(self.ROOT_DIR, "img")
        self.IMG_TRAIN_CONTENT_DIR = os.path.join(self.IMG_DIR, "train_content")
        self.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "train2017")
        self.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "test2017")
        self.IMG_TEST_CONTENT_DIR = os.path.join(self.IMG_DIR, "test_content")
        self.IMG_TEST_RESULTS_DIR = os.path.join(self.IMG_DIR, "test_results")
        self.IMG_STYLE_DIR = os.path.join(self.IMG_DIR, "styles")
        self.LIB_DIR = os.path.join(self.ROOT_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "chkpts") 
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")
    # end
# end

class train_utils:
    def __init__(self):
        self.module_paths = module_paths()
        self.filenames = self.get_image_filenames()
        self.images = self._load_training_data(resolution=[256, 256])
    # end

    def get_image_filenames(self):
            jpg_files = [os.path.join(self.module_paths.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR, file) 
                    for file in os.listdir(self.module_paths.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR)
                    if file.endswith(".jpg")]
            return jpg_files 
    # end

    def _load_training_data(self, resolution):
        target_height, target_width = resolution
        with tf.name_scope("load_MSCOCO_data"):
            filename_queue = tf.train.string_input_producer(jpg_files)
            image_reader = tf.WholeFileReader() # intialize file reader
            _, image_file = image_reader.read(filename_queue) # return filename and pixel values as tensor
            image = tf.image.decode_jpeg(image_file, channels=3)
            resized_image = tf.image.resize_images(images=image, 
                                                size=[target_height, target_width])
            resized_image_squeeze = tf.squeeze(resized_image)
            return resized_image_squeeze
    # end

    def get_batch(self, image):
        batch_results, _ = tf.train.shuffle_batch([self.images, _], 
                batch_size=32,
                num_threads=4,
                capacity=10000,
                min_after_dequeue=5000)
        return batch_results
    # end
# end
