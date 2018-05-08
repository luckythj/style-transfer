from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import argparse
import transformnet, numpy as np, vgg, pdb, os
import tensorflow as tf
from image_utils import get_img, save_img


DEVICE = '/gpu:0'
INPUT_IMAGE = 'data/test.jpg'
OUTPUT_IMAGE = 'data/test_out.jpg'
CHECKPOINT_DIR = 'data/checkpoints/'
IMAGE_SIZE = 256

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', 
                        required=True)

    parser.add_argument('--input_image', type=str,
                        dest='input_image',help='dir or file to transform',
                        metavar='INPUT_IMAGE', 
                        default=INPUT_IMAGE)

    parser.add_argument('--output_image', type=str,
                        dest='output_image', 
                        help='destination (dir or file) of transformed file or files', 
                        metavar='OUTPUT_IMAGE',
                        default=OUTPUT_IMAGE)

    parser.add_argument('--image_size', type=str,
                        dest='image_size',help='dir or file to transform',
                        metavar='IMAGE_SIZE', 
                        default=IMAGE_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser


def run_image(input_image, output_image, checkpoint_dir):

    image = get_img(input_image, (128, 128, 3))
    image_shape = image.shape

    print(image_shape)

    with tf.Graph().as_default(), tf.Session() as sess:

        X_image = tf.placeholder(tf.float32, shape=(1,)+image_shape, name='X_image')
        T_image = transformnet.net(X_image)

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found!")

        else:
            saver.restore(sess, checkpoint_dir)

        T_image = sess.run(T_image, feed_dict={X_image: [image]})
        save_img(output_image, T_image[0])

def main():
    # parser = build_parser()
    # opts = parser.parse_args()
    # check_opts(opts)
    run_image(INPUT_IMAGE, OUTPUT_IMAGE, CHECKPOINT_DIR)


if __name__ == '__main__':
    main()






