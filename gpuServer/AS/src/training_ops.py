import tensorflow as tf
from module_utils import module_paths, train_utils

def train():
    with tf.Graph().as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            train_manager = train_utils()
            coordinator = tf.train.Coordinator() # use coordinator to manage loading image files
            threads = tf.train.start_queue_runners(coord=coordinator)

            with tf.name_scope("get_batches"):
                for _ in range((round(len(train_manager.filenames)/32))):
                    result = sess.run([train_manager.get_batch(batch_results)])
                    print(result[0])
                    print(result[0].shape)
                    exit(0)

            # kill coordinator and join threads
            coordinator.request_stop() 
            coordinator.join(threads)
# end 

train()
