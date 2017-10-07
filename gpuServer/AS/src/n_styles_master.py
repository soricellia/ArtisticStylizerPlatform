import tensorflow as tf
import module_utils

tf.flags.DEFINE_string("run_type", "train", "run type for module")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
