from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

def gauss_l(a=0.4):
    kernel_1d = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2]
    kernel_3d = np.zeros((5, 1, 3, 3), "float32")
    kernel_3d[:,0,0,0] = kernel_1d
    kernel_3d[:,0,1,1] = kernel_1d
    kernel_3d[:,0,2,2] = kernel_1d
    def fn(x):
        return K.conv2d(K.conv2d(x, kernel_3d, strides=(2,1)), K.permute_dimensions(kernel_3d, (1, 0, 2, 3)), strides=(1,2))
    return Lambda(fn)

def build(octaves=1,a=0.4):
    image_input = Input((None,None,3))
    gaussian_pyramid = [image_input]
    for _ in range(octaves-1):
        level = gauss_l()(gaussian_pyramid[-1])
        gaussian_pyramid.append(level)
    return Model(inputs=image_input, outputs=gaussian_pyramid)
