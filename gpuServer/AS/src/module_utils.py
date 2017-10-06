import os

class module_paths:
    def __init__(self):
        self.SRC_DIR = os.path.abspath(os.curdir)
        self.ROOT_DIR = "/".join(SRC_DIR.split("/")[:-1])
        self.IMG_DIR = os.path.join(ROOT_DIR, "img")
        self.IMG_CONTENT_DIR = os.path.join(IMG_DIR, "content")
        self.IMG_RESULTS_DIR = os.path.join(IMG_DIR, "results")
        self.IMG_STYLE_DIR = os.path.join(IMG_DIR, "style")
        self.LIB_DIR = os.path.join(ROOT_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(ROOT_DIR, "chkpts") 
        self.LOGS_DIR = os.path.join(LIB_DIR, "logs")
    # end
# end


