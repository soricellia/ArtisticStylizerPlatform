import tensorflow as tf
from scipy import misc
import numpy as np
import os

class module_utils:
    def __init__(self, FLAGS):
        
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

        #self.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "train2017")
        self.IMG_TRAIN_MSCOCO_TRAIN2017_CONTENT_DIR = os.path.join(self.IMG_TRAIN_CONTENT_DIR, "trainMicro")

        #self.IMG_TRAIN_WIKIART_STYLE_DIR = os.path.join(self.IMG_TRAIN_STYLE_DIR, "train")
        self.IMG_TRAIN_WIKIART_STYLE_DIR = os.path.join(self.IMG_TRAIN_STYLE_DIR, "trainMicro")
        
        # path to images used for model inference
        self.IMG_TEST_CONTENT_DIR = os.path.join(self.IMG_DIR, "test_content")
        self.IMG_TEST_RESULTS_DIR = os.path.join(self.IMG_DIR, "test_results")
        self.IMG_STYLE_DIR = os.path.join(self.IMG_DIR, "styles")
        self.IMG_STYLE_CONTRIB_DIR = os.path.join(self.IMG_DIR, "style_contrib")

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

    def load_data(self, dir):
        jpg_files = [os.path.join(dir, file) 
                    for file in os.listdir(dir)
                    if file.endswith(".jpg")]

        loaded_jpg_files = []
        for jpg_file in jpg_files:
            loaded_jpg_file = misc.imresize(misc.imread(jpg_file), 
                                            size=[self.img_height_dim, self.img_width_dim], 
                                            interp='bilinear') 
            if loaded_jpg_file.shape == (256,256,3):
                loaded_jpg_files.append(loaded_jpg_file)
        all_jpg_images = np.stack(loaded_jpg_files)
        return all_jpg_images 
    # end

    def get_batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batch_per_epoch = int((len(data)-1) / batch_size)+1
        for epoch in range(num_epochs):
            # shuffle the data at each epoch
            if shuffle:
                shuffle_indicies = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffled_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num+1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    # end
# end
