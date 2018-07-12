from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from keras.layers import Input, Lambda, Multiply
from keras import backend as K
import tensorflow as tf

def gram_l(offset):
    def fn(x):
        x1 = x[0] + offset
        x2 = x[1] + offset
        shape = K.shape(x1)
        x2 = tf.image.resize_images(x2,shape[1:3])
        shape2 = K.shape(x2)
        x1 = K.permute_dimensions(x1,(1,2,3,0))
        x1 = tf.image.resize_images(x1, shape2[2:])
        x1 = K.permute_dimensions(x1,(3,0,1,2))

        x1 = K.reshape(x1, (-1, shape[3]))
        x2 = K.reshape(x2, (-1, shape[3]))
        x = K.dot(K.transpose(x1),x2)

        shape = K.cast(shape[1:],"float32")
        shape2 = K.cast(shape2[1:],"float32")

        shape = (2 * shape[0]*shape[1]*shape2[2])
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
    new_outs = []
    for i in range(len(vgg.outputs)-1):
        l1 = vgg.outputs[i]
        l2 = vgg.outputs[i+1]
        g = gram_layer([l1,l2])
        new_outs.append(g)
    model = Model(vgg.inputs, new_outs)
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
