from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import argparse
import transformnet, numpy as np, vgg, pdb, os
import tensorflow as tf
from image_utils import get_img, save_img

INPUT_IMAGE = 'data/test.jpg'
OUTPUT_PATH = 'data/test_out.jpg'
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
                        dest='input_image',help='input file to transform',
                        metavar='INPUT_IMAGE',
                        required=True) 

    parser.add_argument('--output_path', type=str,
                        dest='output_path', 
                        help='destination (dir or file) of transformed file', 
                        metavar='OUTPUT_PATH',
                        required=True)

    parser.add_argument('--image_size', type=int,
                        dest='image_size',help='size of the transformed image',
                        metavar='IMAGE_SIZE', 
                        default=IMAGE_SIZE)

    return parser

def check_opts(opts):
    # assert os.path.exists(opts.checkpoint_dir), 'Checkpoint not found!'
    assert os.path.exists(opts.input_image), 'Input image not found!'
    if os.path.isdir(opts.output_path):
        exists(opts.output_path, 'Output dir not found!')

def transform_image(input_image, output_path, checkpoint_dir, image_size):
    image = get_img(input_image, (image_size, image_size, 3))
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
        save_img(output_path, T_image[0])

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)
    kwargs = {
        "checkpoint_dir":opts.checkpoint_dir,
        "input_image":opts.input_image,
        "output_path":opts.output_path,
        "image_size":opts.image_size,
    }

    transform_image(**kwargs)

if __name__ == '__main__':
    main()






