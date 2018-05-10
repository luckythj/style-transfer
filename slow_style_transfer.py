import sys, os, pdb, argparse
sys.path.insert(0, 'src')
import tensorflow as tf
import numpy as np
from image_utils import load_image, preprocess_image, deprocess_image, save_img
from squeezenet import SqueezeNet
from loss import content_loss, style_loss, tv_loss, gram_matrix
import matplotlib.pyplot as plt


# squeezenet
STYLE_LAYERS = [1, 4, 6, 7]
CONTENT_LAYER = 3
STYLE_LAYER_WEIGHTS = [300000, 1000, 15, 3]
STYLE_WEIGHT = 1
CONTENT_WEIGHT = 6e-2
TV_WEIGHT = 2e-2
STYLE_SIZE = 256
IMAGE_SIZE = 256
MAX_ITERATIONS = 200

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, output_path, max_iterations=200, init_random = False):
    """Run style transfer!
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """

    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = model.extract_features(model.image)
    content_target = sess.run(feats[content_layer],
                              {model.image: content_img[None]})

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    style_feat_vars = [feats[idx] for idx in style_layers]
    style_target_vars = []
    # Compute list of TensorFlow Gram matrices
    for style_feat_var in style_feat_vars:
        style_target_vars.append(gram_matrix(style_feat_var))
    # Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
    style_targets = sess.run(style_target_vars, {model.image: style_img[None]})

    # Initialize generated image to content image
    
    if init_random:
        img_var = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
    else:
        img_var = tf.Variable(content_img[None], name="image")

    # Extract features on generated image
    feats = model.extract_features(img_var)
    # Compute loss
    c_loss = content_loss(content_weight, feats[content_layer], content_target)
    s_loss = style_loss(feats, style_layers, style_targets, style_weights)
    t_loss = tv_loss(img_var, tv_weight)
    loss = c_loss + s_loss + t_loss
    
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    max_iter = max_iterations

    # Create and initialize the Adam optimizer
    lr_var = tf.Variable(initial_lr, name="lr")
    # Create train_op that updates the generated image when run
    with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
    # Initialize the generated image and optimization variables
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
    sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))
    # Create an op that will clamp the image values when run
    clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))
    
    # f, axarr = plt.subplots(1,2)
    # axarr[0].axis('off')
    # axarr[1].axis('off')
    # axarr[0].set_title('Content Source Img.')
    # axarr[1].set_title('Style Source Img.')
    # axarr[0].imshow(deprocess_image(content_img))
    # axarr[1].imshow(deprocess_image(style_img))
    # plt.show()
    # plt.figure()
    
    # Hardcoded handcrafted 
    for t in range(max_iter):
        # Take an optimization step to update img_var
        sess.run(train_op)
        if t < decay_lr_at:
            sess.run(clamp_image_op)
        if t == decay_lr_at:
            sess.run(tf.assign(lr_var, decayed_lr))
        if t % 100 == 0:
            print('Iteration {}'.format(t))
            # img = sess.run(img_var)
            # plt.imshow(deprocess_image(img[0], rescale=True))
            # plt.axis('off')
            # plt.show()
    print('Iteration {}'.format(t))
    img = sess.run(img_var)        
    # plt.imshow(deprocess_image(img[0], rescale=True))
    # plt.axis('off')
    # plt.show()
    save_img(output_path, deprocess_image(img[0], rescale=True))


tf.reset_default_graph() # remove all existing variables in the graph 
sess = get_session() # start a new Session

# Load pretrained SqueezeNet model
SAVE_PATH = 'data/squeezenet.ckpt'
if not os.path.exists(SAVE_PATH):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

# Load data for testing
content_img_test = preprocess_image(load_image('styles/tubingen.jpg', size=192))[None]
style_img_test = preprocess_image(load_image('styles/starry_night.jpg', size=192))[None]
# answers = np.load('style-transfer-checks-tf.npz')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str,
                    dest='style_image', help='style image path',
                    metavar='STYLE', required=True)
    parser.add_argument('--style_size', type=int,
                        dest='style_size',help='size of the style image',
                        metavar='STYLE_SIZE', 
                        default=STYLE_SIZE)
    parser.add_argument('--input_image', type=str,
                        dest='input_image',help='input file to transform',
                        metavar='INPUT_IMAGE',
                        required=True) 
    parser.add_argument('--image_size', type=int,
                        dest='image_size',help='size of the transformed image',
                        metavar='IMAGE_SIZE', 
                        default=IMAGE_SIZE)
    parser.add_argument('--content_weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style_weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--tv_weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--iterations', type=int,
                        dest='max_iterations',
                        help='optimization iterations (default %(default)s)',
                        metavar='MAX_ITERATIONS', default=MAX_ITERATIONS)
    parser.add_argument('--output_path', type=str,
                        dest='output_path', 
                        help='destination (dir or file) of transformed file', 
                        metavar='OUTPUT_PATH',
                        required=True)

    return parser

def check_opts(opts):
    assert os.path.exists(opts.style_image), 'Style image not found!'
    assert os.path.exists(opts.input_image), 'Input image not found!'
    if os.path.isdir(opts.output_path):
        exists(opts.output_path, 'Output dir not found!')
def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)
    kwargs = {
        'content_image' : opts.input_image,
        'style_image' : opts.style_image,
        'image_size' : opts.image_size,
        'style_size' : opts.style_size,
        'content_layer' : CONTENT_LAYER,
        'content_weight' : opts.content_weight,
        'style_layers' : STYLE_LAYERS,
        'style_weights' : np.array(STYLE_LAYER_WEIGHTS) * opts.style_weight,
        'tv_weight' : opts.tv_weight,
        'max_iterations': opts.max_iterations,
        'output_path': opts.output_path,
        'init_random': True, # we want to initialize our image to be random
    }
    style_transfer(**kwargs)


if __name__ == "__main__":
    main()




