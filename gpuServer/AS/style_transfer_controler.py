import tensorflow as tf


tf.flags.DEFINE_string("run_type", "train", "run type for module")
tf.flags.DEFINE_integer("img_dim", 256, "image dimesions aka resolution")

# training configs
tf.flags.DEFINE_integer("num_epochs", 5, "number of training epochs")
tf.flags.DEFINE_integer("chkpt_every", 100, "save every (default:100) steps")
tf.flags.DEFINE_integer("num_chkpt", 2, "number of checkpoints to save")
tf.flags.DEFINE_integer("batch_size", 20, "batch size (default: 20)")

# training optimization config
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.flags.DEFINE_float("momentum_2", 0.999, "momentum_2")

# n-styles loss weighting factors
tf.flags.DEFINE_float("weighting_factor_content_loss", 1.5e0, "contribution of content loss")
tf.flags.DEFINE_float("weighting_factor_style_loss", 1e2, "contribution of style loss")
tf.flags.DEFINE_float("weighting_factor_tv_loss", 2e2, "contribution of tv loss")

# testing configs
#tf.flags.DEFINE_string("style_contribution", "", "run type for module")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
