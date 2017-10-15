import tensorflow as tf
import threading
import training_ops as ops

# training options
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.85, "faction of gpu memory permitted to use")
tf.flags.DEFINE_integer("num_epochs", 5, "number of training epochs")
tf.flags.DEFINE_boolean("resume", False, "resume training from most recent checkpoint")
tf.flags.DEFINE_integer("chkpt_every", 100, "save every (default:100) steps")
tf.flags.DEFINE_integer("num_chkpt", 2, "number of checkpoints to save")

# preprocessing options
tf.flags.DEFINE_integer("final_size", 256, "size of image used for training")
tf.flags.DEFINE_integer("transient_size", 512, "size images before cropping,")

# pastiche options
tf.flags.DEFINE_float("weighting_factor_content_loss", 1, "contribution of content loss")
tf.flags.DEFINE_float("weighting_factor_style_loss", 1e2, "contribution of style loss")
tf.flags.DEFINE_float("weighting_factor_tv_loss", 0, "contribution of tv loss")
tf.flags.DEFINE_string("target_content_layer", "relu4_1", "target content layer used "
                                                          "to compute loss")
tf.flags.DEFINE_string("target_style_layers", "relu1_1 relu2_1 relu3_1 relu4_1",
                       "target style layers used to compute the loss, as space separated list")

# model options
tf.flags.DEFINE_integer("batch_size", 8, "batch size (default: 8)")
tf.flags.DEFINE_string("activation", "relu", "activation function in the decoder")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer used")
tf.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
tf.flags.DEFINE_float("learning_rate_decay", 5e-5, "learning rate decay")
tf.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.flags.DEFINE_float("weight_decay", 0, "weight decay")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# TODO set option for FLAGS.resume
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options)
    sess = tf.Session(config=session_config)
    with sess.as_default():

        # set up project constants
        train_ops = ops.training_ops(FLAGS)

        """
        # checkpoint model TODO look up tf.global_variables() see denny britz's code
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_chkpt)
        # model summaries
        summary_writer = tf.summary.FileWriter(train_ops.mod_utils.LOGS_DIR, sess.graph)
        # global step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        """

        enqueue_thread = threading.Thread(target=train_ops.enqueue, args=[sess])
        enqueue_thread.isDaemon()
        enqueue_thread.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.num_epochs):
            content_batch, style_batch = sess.run([train_ops.content_batch_op, train_ops.style_batch_op])
            print(content_batch)
            print(style_batch)

