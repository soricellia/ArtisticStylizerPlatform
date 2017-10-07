import os

class module_paths:
    def __init__(self):
        self.SRC_DIR = os.path.abspath(os.curdir)
        self.ROOT_DIR = "/".join(SRC_DIR.split("/")[:-1])
        self.IMG_DIR = os.path.join(ROOT_DIR, "img")
        self.IMG_TRAIN_CONTENT_DIR = os.path.join(IMG_DIR, "train_content")
        self.IMG_TEST_CONTENT_DIR = os.path.join(IMG_DIR, "test_content")
        self.IMG_TEST_RESULTS_DIR = os.path.join(IMG_DIR, "test_results")
        self.IMG_STYLE_DIR = os.path.join(IMG_DIR, "styles")
        self.LIB_DIR = os.path.join(ROOT_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(ROOT_DIR, "chkpts") 
        self.LOGS_DIR = os.path.join(LIB_DIR, "logs")
    # end
# end


