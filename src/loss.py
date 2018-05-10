import tensorflow as tf

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [N, height, width, channels]
    - content_target: features of the content image, Tensor with shape [N, height, width, channels]
    
    Returns:
    - scalar content loss [N,]
    """
    shapes = tf.shape(content_current)
    N,H,W,C = shapes[0], shapes[1], shapes[2], shapes[3] 
    
    loss = content_weight * tf.reduce_sum(tf.square(content_current - content_original), axis=[1,2,3]) 
    loss /= tf.cast(H*W*C, tf.float32) # normalize it by default
    
    return loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (N, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (N, C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    shapes = tf.shape(features)
    N,H,W,C = shapes[0], shapes[1], shapes[2], shapes[3]
    
    # Reshape feature map from [N, H, W, C] to [N, H*W, C].
    F_l = tf.reshape(features, shape=[N, H*W, C])

    # Transpose from [N, H*W, C] to [N, C, H*W]
    F_l_T = tf.transpose(F_l, perm=[0, 2, 1])
     
    # Gram calculation is just a matrix multiply 
    # of F_l and F_l transpose to get [C, C] output shape.
    gram = tf.matmul(F_l_T,F_l) # [N, C, C]
    
    if normalize == True:
        gram /= tf.cast(H*W*C, tf.float32)
    
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function. each feat's shape: [N, H, W, C].
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss. 
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor contataining the scalar style loss. [N,]
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.

    # Initialise style loss to 0.0
    N = feats[style_layers[0]].shape[0]
    style_loss = tf.zeros(N)
    
    # Compute style loss for each desired feature layer and then sum.
    for i in range(len(style_layers)):
        style_gram_l = style_targets[i] # [1, C, C]
        cur_gram_l = gram_matrix(feats[style_layers[i]]) # [N, C, C]
        style_loss_l = style_weights[i] * tf.reduce_sum(tf.square(cur_gram_l - style_gram_l), axis=[1,2])
        style_loss_l /= style_gram_l.size # normalize it
        style_loss += style_loss_l
    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (N, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight. [N,]
    """
    # Your implementation should be vectorized and not require any loops!  
    shapes = tf.shape(img)
    N,H,W,C = shapes[0], shapes[1], shapes[2], shapes[3]

    w_variance = tf.reduce_sum(tf.square(img[:,:,1:,:] - img[:,:,:-1,:]), axis=[1,2,3]) #[N, H, W-1, C] -> [N,]
    w_variance /= tf.cast(H*(W-1)*C, tf.float32) 

    h_variance = tf.reduce_sum(tf.square(img[:,1:,:,:] - img[:,:-1,:,:]), axis=[1,2,3]) #[N, H-1, W, C] -> [N,]
    h_variance /= tf.cast((H-1)*W*C, tf.float32) 

    loss = tv_weight * (w_variance + h_variance)

    return loss