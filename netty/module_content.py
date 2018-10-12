from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from keras.layers import Input, Lambda
from keras import backend as K

def loss_l(w):
    def fn(x):
        x = K.sum(K.square(x[0]-x[1])) * w / 2
        return K.expand_dims(x)
    return Lambda(fn)

def content_l():
    def fn(x):
        return x
    return Lambda(fn)

def build(args):
    vgg = model_vgg.build(args)
    vgg = extract_layers(vgg, args["content_layers"])
    octave_model = model_octave.build(1, args["octave_a"])
    model = attach_models(octave_model, vgg)
    content_layer = content_l();
    model = attach_models(model, content_layer)

    targets = []
    losses = []
    for i in range(len(model.outputs)):
        targets.append(Input(model.outputs[i].shape[1:]))
        layer_weight = 1
        layer_loss = loss_l(layer_weight)([targets[i], model.outputs[i]])
        losses.append(layer_loss)
    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) / len(losses) * args["content_w"])(losses)

    loss_model = Model(model.inputs + targets, loss)

    return loss_model, model, targets
