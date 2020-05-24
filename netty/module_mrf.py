from netty.build_utils import *
from netty.vgg_utils import *
from netty import model_vgg
from netty import model_octave
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

def make_patches(patch_size, patch_stride, input_shape):
    def fn(x):
        x = K.reshape(x, shape=(1,input_shape[0],input_shape[1],input_shape[2]))
        x = K.permute_dimensions(x,(3,1,2,0))
        patches = tf.extract_image_patches(
            images = x,
            ksizes = [1,patch_size,patch_size,1],
            strides = [1,patch_stride,patch_stride,1],
            rates = [1,1,1,1],
            padding = "VALID"
        )
        pshape = K.shape(patches)
        patches = K.reshape(patches, shape=(pshape[0],pshape[1]*pshape[2],patch_size,patch_size))
        patches = K.permute_dimensions(patches,(1,2,3,0))
        patches = K.expand_dims(patches, axis=0)
        # patches = tf.stop_gradient(patches)
        return patches
    return Lambda(fn)

def match_patches():
    def fn(args):
        # [yx,py,px,ch]
        x = args[1][0]
        y = args[0][0]
        y = K.permute_dimensions(y, (1,2,3,0))
        convs = K.conv2d(x, y, padding="valid")
        argmax = K.argmax(convs, axis=0)
        return K.expand_dims(argmax, axis=0)
    return Lambda(fn)

def repatch():
    def fn(args):
        best = K.cast(args[1][0][0][0], "int32")
        x = args[0][0]
        x = tf.gather(x, best)
        return K.expand_dims(x, axis=0)
    return Lambda(fn)

def loss_l():
    def fn(args):
        x = args[0]
        y = args[1]
        shape = K.shape(x)
        loss = K.sum(K.square(y - x))/K.cast(shape[0]*shape[1]*shape[2]*shape[3], "float32")
        return K.expand_dims(loss, axis=0)
    return Lambda(fn)

def mrf_m(ks,s,l,o,mix_shape,tar_shape,maps=None):
    mix = Input((None,None,None))
    tar = Input((None,None,None))

    tgs = [tar]

    mix_shape = get_vgg_shape(mix_shape, l, o)
    tar_shape = get_vgg_shape(tar_shape, l, o)

    mix_p = make_patches(ks,s,mix_shape)(mix)
    tar_p = make_patches(ks,s,tar_shape)(tar)
    match = match_patches()([mix_p,tar_p])
    repatched = repatch()([tar_p,match])

    loss = loss_l()([mix_p,repatched])

    model = Model([mix,tar],loss)
    return model

def build(args):
    vgg = model_vgg.build(args)
    vgg = extract_layers(vgg, args["mrf_layers"])
    octave_model = model_octave.build(args["octaves"], args["octave_a"])
    model = attach_models(octave_model, vgg)

    targets = []
    losses = []
    layers_num = len(model.outputs) // args["octaves"]
    mix_shape = [args["size"][1],args["size"][0]]
    tar_shape = args["style_shape"]
    for o in range(args["octaves"]):
        for l in range(layers_num):
            i = layers_num * o + l

            targets.append(Input(model.outputs[i].shape[1:]))
            layer_weight = 1 / len(args["style_layers"])
            mrf_model = mrf_m(args["mrf_patch_size"], args["mrf_patch_stride"], args["mrf_layers"][l], o, mix_shape, tar_shape)
            layer_loss = mrf_model([model.outputs[i], targets[i]])
            layer_loss = Lambda(lambda x:x*layer_weight)(layer_loss)
            losses.append(layer_loss)

    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) * args["mrf_w"])(losses)

    loss_model = Model(model.inputs + targets, loss)
    return loss_model, model, targets
