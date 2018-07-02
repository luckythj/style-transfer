import tensorflow as tf, numpy as np, os
import transformnet
import vgg
import math
from squeezenet import SqueezeNet
from image_utils import load_image, get_img
from loss import style_loss, content_loss, tv_loss, gram_matrix

# vgg net
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
STYLE_WEIGHTS = [1, 1, 1, 1, 1]
# STYLE_WEIGHTS = (300000, 1000, 15, 3, 1)
CONTENT_LAYER = 'relu4_2'

STYLE_SIZE = 256
IMAGE_SIZE = 256
CHANNEL_SIZE = 3

def optimize(content_paths, style_path, 
        content_weight, style_weight, tv_weight,
        vgg_path, epochs=2, batch_size=4,
        print_iterations=1, checkout_iterations=10000,
        checkpoint_dir='data/checkpoints/', learning_rate=1e-3):

    # style_img = load_image(style_path, size=STYLE_SIZE)
    style_img = get_img(style_path, [STYLE_SIZE,STYLE_SIZE,CHANNEL_SIZE])
    style_shape = (1, ) + style_img.shape
    content_shape = (batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE)

    style_target_features = []
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        # extract style features of style target
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_feat_vars = [net[layer] for layer in STYLE_LAYERS]
        for style_feat_var in style_feat_vars:
            style_target_features.append(gram_matrix(style_feat_var))
        style_target_features = sess.run(style_target_features, feed_dict={style_image: style_img[None]})

    with tf.Graph().as_default(), tf.Session() as sess:
        # extract feature map of content target
        X_content = tf.placeholder(tf.float32, shape=content_shape, name='X_content')
        X_content_pre = vgg.preprocess(X_content)
        feats = vgg.net(vgg_path, X_content_pre)
        X_content_target_feature = feats[CONTENT_LAYER]

        # transformed X_content to T_content
        X_content_pre = transformnet.preprocess(X_content)
        T_content = transformnet.net(X_content_pre)

        # extract feature map of T_content
        T_content_pre = vgg.preprocess(T_content)
        feats = vgg.net(vgg_path, T_content_pre)

        # loss

        s_loss = style_loss(feats, STYLE_LAYERS, style_target_features, np.array(STYLE_WEIGHTS) * style_weight)
        c_loss = content_loss(content_weight, feats[CONTENT_LAYER], X_content_target_feature)
        t_loss = tv_loss(T_content_pre, tv_weight)

        s_loss = tf.reduce_sum(s_loss)/batch_size 
        c_loss = tf.reduce_sum(c_loss)/batch_size
        t_loss = tf.reduce_sum(t_loss)/batch_size

        loss = s_loss + c_loss + t_loss

        # train
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        iters_per_epoch = math.floor(len(content_paths)/batch_size)
        for epoch in range(epochs):
            for iteration in range(iters_per_epoch):
                batch_start = batch_size * iteration
                batch_end = batch_start + batch_size
                # TODO need to shuffle data for each epoch !!
                X_batch = np.zeros(content_shape, dtype=np.float32)
                for i, img_path in enumerate(content_paths[batch_start:batch_end]):
                    X_batch[i] = get_img(img_path, [IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE])
                    
                assert len(X_batch) == batch_size
                train_step.run(feed_dict={X_content: X_batch})

                is_last_iter = epoch == epochs-1 and iteration == iters_per_epoch-1

                # print
                if iteration % print_iterations == 0 or is_last_iter:
                    _T_content, _loss, _s_loss, _c_loss, _t_loss, _feats = sess.run([T_content, loss, s_loss, c_loss, t_loss, feats], feed_dict={X_content: X_batch})
                    print('Epoch %d, iteration %d: loss %s, s_loss %s, c_loss %s, t_loss %s'%(epoch, iteration, _loss, _s_loss, _c_loss, _t_loss))
        
                # save checkpoint
                if iteration % checkout_iterations == 0 or is_last_iter:
                    saver = tf.train.Saver()
                    ckpt_suffix = '_final.ckpt' if is_last_iter else '_epoch%d_iter%d.ckpt'%(epoch, iteration)
                    saver.save(sess, os.path.join(checkpoint_dir,'checkpoint'+ckpt_suffix))

