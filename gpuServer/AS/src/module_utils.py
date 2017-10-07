import tensorflow as tf
import os

class module_paths:
    def __init__(self):
        self.SRC_DIR = os.path.abspath(os.curdir)
        self.ROOT_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.IMG_DIR = os.path.join(self.ROOT_DIR, "img")
        self.IMG_TRAIN_CONTENT_DIR = os.path.join(self.IMG_DIR, "train_content")
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
        self._get_training_data()
    # end
    def _get_training_data(self):
        filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once("{}/*.jpg".format(self.module_paths.IMG_TRAIN_CONTENT_DIR)))
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator)

            image_tensor = sess.run([image])
            print(image_tensor)

            coord.request_stop()
            coord.join(threads)



    # end
    #def get_batch(self):
    # end
# end


if __name__ == "__main__":
    train_utils()

