import tensorflow as tf
from scipy import misc
import numpy as np
import os

class module_utils:
    def __init__(self, FLAGS):
        self.init_prj_constants(FLAGS)
    # end

    def init_prj_constants(self, FLAGS):
        # project paths
        self.SRC_DIR = os.path.abspath(os.curdir)
        self.ROOT_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.LIB_DIR = os.path.join(self.ROOT_DIR, "lib")
        self.IMG_DIR = os.path.join(self.ROOT_DIR, "img")

        # path to persisted models and model logs
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "chkpts") 
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

        # paths to images needed for training
        self.IMG_TRAIN_CONTENT_DIR = os.path.join(self.IMG_DIR, "train_content")
        self.IMG_TRAIN_STYLE_DIR = os.path.join(self.IMG_DIR, "train_style")

        self.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "train2017")
        #self.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "trainMicro")

        self.IMG_TRAIN_WIKIART_STYLE_DIR = os.path.join(self.IMG_TRAIN_STYLE_DIR, "train")
        #self.IMG_TRAIN_WIKIART_STYLE_DIR = os.path.join(self.IMG_TRAIN_STYLE_DIR, "trainMicro")
        
        # path to images used for model inference
        self.IMG_TEST_CONTENT_DIR = os.path.join(self.IMG_DIR, "test_content")
        self.IMG_TEST_RESULTS_DIR = os.path.join(self.IMG_DIR, "test_results")
        self.IMG_STYLE_DIR = os.path.join(self.IMG_DIR, "styles")
        self.IMG_STYLE_CONTRIB_DIR = os.path.join(self.IMG_DIR, "style_contrib")

        # FLAGS
        self.num_epochs = FLAGS.num_epochs
        self.resume = FLAGS.resume
        self.chkpt_every = FLAGS.chkpt_every
        self.num_chkpt = FLAGS.num_chkpt
        self.final_size = FLAGS.final_size
        self.transient_size = FLAGS.transient_size
        self.weighting_factor_content_loss = FLAGS.weighting_factor_content_loss
        self.weighting_factor_style_loss = FLAGS.weighting_factor_style_loss
        self.weighting_factor_tv_loss = FLAGS.weighting_factor_tv_loss
        self.target_content_layer = FLAGS.target_content_layer
        self.target_style_layers = FLAGS.target_style_layers.split(" ")
        self.batch_size = FLAGS.batch_size
        self.activation = FLAGS.activation
        self.optimizer = FLAGS.optimizer
        self.learning_rate = FLAGS.learning_rate
        self.learning_rate_decay = FLAGS.learning_rate_decay
        self.momentum = FLAGS.momentum
        self.weight_decay = FLAGS.weight_decay

        self.batch_shape = (self.batch_size, self.final_size, self.final_size, 3)
    # end

    def load_data(self, jpg_file):
        """
        desc: this function loads all jpg images in the specified directory performing all necessary
              preprocessing (resizing and center cropping) 
              returning an array [num_examples, height, width, channel] to be used get_batch_iter function
              to generate batches during model training
        :param dir: directory where jpg images are located
        :return: returning an array [num_examples, height, width, channel]
        """

        def center_crop(img):
            """
            desc: center crops an image
            :param img: original image
            :return: center cropped image
            """
            y, x, _ = img.shape
            startx = x // 2 - (self.final_size// 2)
            starty = y // 2 - (self.final_size // 2)
            return img[starty:starty + self.final_size, startx:startx + self.final_size]
        # end

        def resize_preserving_aspect_ratio(img):
            """
            desc: resize smallest dimension of image to target size while preserving aspect ratio
            :return: 
            """
            height, width = img.shape[0], img.shape[1]
            get_ratio = lambda s: s / self.transient_size
            if height < width:
                resized_shape = (self.transient_size, round(width / get_ratio(height)), 3)
            else:
                resized_shape = (round(height / get_ratio(width)), self.transient_size, 3)
            return misc.imresize(img, resized_shape)  # uses bilinear interpolation
        # end

        # load, resize, and center crop all jpg files
        loaded_jpg_file = misc.imread(jpg_file)

        if(len(loaded_jpg_file.shape) == 3):
            resized_jpg = center_crop(resize_preserving_aspect_ratio(loaded_jpg_file))
            assert(resized_jpg.shape == (self.final_size, self.final_size, 3)), "resized image is " \
                                                                                "not correct dimensions"
            return resized_jpg
    # end

    def get_batch(self, dir):
        # get all jpg file names in specified directory
        jpg_files = np.asarray([os.path.join(dir, file) for file in os.listdir(dir)])

        while True:
            batch = np.zeros(self.batch_shape, dtype=np.float32)
            i = 0
            while i < self.batch_size:
                try:
                    batch[i] = self.load_data(jpg_file=np.random.choice(jpg_files))
                except Exception as e:
                    print(e)  # TODO log this
                    continue
                i += 1
            yield batch
    # end
# end
