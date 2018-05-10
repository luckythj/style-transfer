import sys, os, pdb
sys.path.insert(0, 'src')
import argparse
from solver import optimize
from image_utils import get_files

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'data/checkpoints/'
CHECKPOINT_ITERATIONS = 10000
PRINT_ITERATIONS = 10
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 20

def build_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', default=CHECKPOINT_DIR)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='print frequency',
                        metavar='PRINT_ITERATIONS',
                        default=PRINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    return parser

def check_opts(opts):
    assert os.path.exists(opts.checkpoint_dir), 'checkpoint dir not found!'
    assert os.path.exists(opts.style), 'style path not found!'
    assert os.path.exists(opts.train_path), 'train path not found!'
    assert os.path.exists(opts.vgg_path), 'vgg network data not found!'
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

def main():
    parser = build_parser()
    options = parser.parse_args()
    content_paths = get_files(options.train_path)
    kwargs = {
        "content_paths": content_paths,
        "style_path": options.style,
        "content_weight": options.content_weight,
        "style_weight": options.style_weight,
        "vgg_path": options.vgg_path,
        "tv_weight": options.tv_weight,
        "epochs": options.epochs,
        "checkout_iterations": options.checkpoint_iterations,
        "print_iterations": options.print_iterations,
        "batch_size": options.batch_size,
        "checkpoint_dir": options.checkpoint_dir,
        "learning_rate": options.learning_rate
    }

    optimize(**kwargs)

if __name__ == "__main__":
    main()