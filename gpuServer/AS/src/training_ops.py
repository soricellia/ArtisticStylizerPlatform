import tensorflow as tf
import module_utils
import os

def train(resolution=[256,256]):
    target_height, target_width = resolution

    with tf.Graph().as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            coordinator = tf.train.Coordinator() # use coordinator to manage loading image files
            threads = tf.train.start_queue_runners(coord=coordinator)

            module_paths = module_utils.module_paths()

            with tf.name_scope("load_data"):
                jpg_files = [os.path.join(module_paths.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR, file) 
                            for file in os.listdir(module_paths.IMG_TRAIN_MSCOCO_TEST2017_CONTENT_DIR)
                            if file.endswith(".jpg")]
                filename_queue = tf.train.string_input_producer(jpg_files )
                image_reader = tf.WholeFileReader() # intialize file reader
                filename, image_file = image_reader.read(filename_queue) # return filename and pixel values as tensor
                image = tf.image.decode_jpeg(image_file, channels=3)
                resized_image = tf.image.resize_images(images=image, 
                                                size=[target_height, target_width])
                resized_image_squeeze = tf.squeeze(resized_image)

            with tf.name_scope("get_batches"):
                for i in range((round(len(jpg_files)/32))):
                    batch_results, _ = tf.train.shuffle_batch([resized_image_squeeze, filename], 
                                                              batch_size=32,
                                                              num_threads=4,
                                                              capacity=10000,
                                                              min_after_dequeue=5000)
                    print(batch_results)


            # kill coordinator and join threads
            coordinator.request_stop() 
            coordinator.join(threads)
# end 

if __name__ == "__main__":
    train()
