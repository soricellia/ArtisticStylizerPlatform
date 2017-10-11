import tensorflow as tf
from scipy import misc
import numpy as np
import os

class module_utils:
    def __init__(self, FLAGS):
        
        # project paths
        self.SRC_DIR = os.path.abspath(os.curdir)
        self.ROOT_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.IMG_DIR = os.path.join(self.ROOT_DIR, "img")
        self.IMG_TRAIN_CONTENT_DIR = os.path.join(self.IMG_DIR, "train_content")
        self.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "train2017")
        #self.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "test2017")
        self.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "trainMicro")
        self.IMG_TEST_CONTENT_DIR = os.path.join(self.IMG_DIR, "test_content")
        self.IMG_TEST_RESULTS_DIR = os.path.join(self.IMG_DIR, "test_results")
        self.IMG_STYLE_DIR = os.path.join(self.IMG_DIR, "styles")
        self.IMG_STYLE_CONTRIB_DIR = os.path.join(self.IMG_DIR, "style_contrib")
        self.LIB_DIR = os.path.join(self.ROOT_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "chkpts") 
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

        # FLAGS
        self.run_type = FLAGS.run_type
        self.process_id = FLAGS.process_id
        self.img_height_dim = FLAGS.img_height_dim
        self.img_width_dim = FLAGS.img_width_dim
        self.num_epochs = FLAGS.num_epochs
        self.chkpt_every = FLAGS.chkpt_every
        self.num_chkpt = FLAGS.num_chkpt
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.momentum = FLAGS.momentum
        self.momentum_2 = FLAGS.momentum_2
        self.weighting_factor_content_loss = FLAGS.weighting_factor_content_loss
        self.weighting_factor_style_loss = FLAGS.weighting_factor_style_loss
        self.weighting_factor_tv_loss = FLAGS.weighting_factor_tv_loss
    # end
# end
