import numpy as np
vgg_offsets = np.asarray([103.939, 116.779, 123.68])
def preprocess(img):
    x = np.copy(img)
    x = x[...,[2,1,0]]
    x = x - vgg_offsets
    return x
def deprocess(x):
    x = np.copy(x) + vgg_offsets
    return x[...,[2,1,0]]
def get_bounds(x0):
    bounds = [[- vgg_offsets[0], 255 - vgg_offsets[0]],
          [- vgg_offsets[1], 255 - vgg_offsets[1]],
          [- vgg_offsets[2], 255 - vgg_offsets[2]]] * (x0.size//3)
    return bounds
