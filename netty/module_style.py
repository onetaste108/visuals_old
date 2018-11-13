from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from keras.layers import Input, Lambda, Multiply
from keras import backend as K
import tensorflow as tf


def gram_l(offset):
    def fn(x):
        x = x + offset
        shape = K.shape(x)
        x = K.reshape(x, (-1, shape[3]))
        x = K.dot(K.transpose(x),x)
        shape = K.cast(shape[1:],"float32")
        shape = (1 * shape[0]*shape[1]*shape[2])
        x = x / shape
        return K.expand_dims(x, axis=0)
    return Lambda(fn)

def loss_l(w):
    def fn(x):
        x = x[0]-x[1]
        x = K.sum(K.square(x)) * w
        return K.expand_dims(x)
    return Lambda(fn)

def build(args):
    vgg = model_vgg.build(args)
    vgg = extract_layers(vgg, args["style_layers"])

    octave_model = model_octave.build(args["octaves"], args["octave_a"])
    gram_layer = gram_l(args["style_offset"])
    model = attach_models(vgg, gram_layer)
    model = attach_models(octave_model, model)

    targets = []
    losses = []
    layers_num = len(model.outputs) // args["octaves"]
    for o in range(args["octaves"]):
        for l in range(layers_num):
            i = layers_num * o + l
            targets.append(Input(model.outputs[i].shape[1:]))
            layer_weight = args["style_lw"][l] / len(args["style_layers"])
            layer_loss = loss_l(layer_weight)([targets[i], model.outputs[i]])
            losses.append(layer_loss)
    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) * args["style_w"])(losses)

    loss_model = Model(model.inputs + targets, loss)
    return loss_model, model, targets
