import tensorflow as tf
from model import MiniNet
import utils
import os


def main():
    # Get the configuration arguments
    args = utils.get_args()
    utils.print_args(args)

    # Allocate a small fraction of GPU and expand the allotted memory as needed
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

    # Essentially defining global variables. TF_CPP_MIN_LOG_LEVEL equates to '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Create a session with config options
    with tf.Session(config=config) as sess:
        # initialize the DNN
        mini = MiniNet(sess, args)

        # Gets all variables that have trainable=True
        model_vars = tf.trainable_variables()

        # slim is a library that makes defining, training, and evaluating NNs simple.
        tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        if args.training == True:
            mini.train()
        else:
            mini.test()


if __name__ == '__main__':
    main()
