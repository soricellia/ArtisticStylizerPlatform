import tensorflow as tf
import threading
import module_utils
import inference_ops
import vgg19
import adaIN
from matplotlib import image
import numpy as np
import os

# training options

tf.flags.DEFINE_string("process_id", "9999", "unique identifier of process")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.1, "faction of gpu memory permitted to use")

# preprocessing options
tf.flags.DEFINE_integer("final_size", 256, "size of image used for training")
tf.flags.DEFINE_integer("transient_size", 512, "size images before cropping,")

# pastiche options
tf.flags.DEFINE_float("weighting_factor_content_loss", 1, "contribution of content loss")
tf.flags.DEFINE_float("weighting_factor_style_loss", 1, "contribution of style loss")
tf.flags.DEFINE_float("weighting_factor_tv_loss", 0, "contribution of tv loss")
tf.flags.DEFINE_string("target_content_layer", "relu4_1", "target content layer used "
                                                          "to compute loss")
tf.flags.DEFINE_string("target_style_layers", "relu1_1 relu2_1 relu3_1 relu4_1",
                       "target style layers used to compute the loss, as space separated list")

# model options
tf.flags.DEFINE_integer("batch_size", 1, "batch size (default: 15)")
tf.flags.DEFINE_string("activation", "relu", "activation function in the decoder")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer used")
tf.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
tf.flags.DEFINE_float("learning_rate_decay", 5e-5, "learning rate decay")
tf.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.flags.DEFINE_float("weight_decay", 0, "weight decay")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# TODO set option for FLAGS.resume
with tf.Graph().as_default() as graph:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options)
    sess = tf.Session(config=session_config)
    with sess.as_default():

        # set up project constants
        mod_utils = module_utils.module_utils(FLAGS, run_type="inference")
        sess.run(tf.global_variables_initializer())

        process_assoc_content_img, process_assoc_style_img = mod_utils.get_inference(process_id=FLAGS.process_id)

        decoder_t7 = "decoder.t7"
        vgg_t7="vgg_normalised.t7"
        alpha=1

        stylized, content_image, style_image = inference_ops.stylize(process_assoc_content_img, process_assoc_style_img, alpha, vgg_t7, decoder_t7, resize=[512,512])
        image.imsave(os.path.join(mod_utils.IMG_INFERENCE_RESULTS_DIR, "{}_results.jpg".format(FLAGS.process_id)), inference_ops.any_to_uint8_clip(stylized))
