from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from keras.layers import Input, Lambda, Multiply
from keras import backend as K

def gram_l():
    def fn(x):
        x = x+100
        shape = K.shape(x)
        x = K.reshape(x, (-1, shape[3]))
        x = K.dot(K.transpose(x),x)
        return K.expand_dims(x, axis=0)
    return Lambda(fn)

def get_shape():
    def fn(x):
        shape = K.shape(x)[1:]
        shape = K.cast(shape,"float32")
        shape = (4 * K.square(shape[0]*shape[1]) * K.square(shape[2]))
        return K.expand_dims(shape)
    return Lambda(fn)

def loss_l():
    def fn(x):
        x = x[0]-x[1]
        x = K.sum(K.square(x))
        return K.expand_dims(x)
    return Lambda(fn)

def build(args):
    vgg = model_vgg.build(args)
    vgg = extract_layers(vgg, args["style_layers"])
    octave_model = model_octave.build(args["octaves"], args["octave_a"])
    vgg_octave = attach_models(octave_model, vgg)
    gram_layer = gram_l()
    model = attach_models(vgg_octave, gram_layer)

    targets = []
    losses = []
    for o in range(args["octaves"]):
        for l in range(len(args["style_layers"])):
            i = len(args["style_layers"]) * o + l
            targets.append(Input(model.outputs[i].shape[1:]))
            layer_weight = 1
            layer_shape = get_shape()(vgg_octave.outputs[i])
            layer_loss = loss_l()([targets[i], model.outputs[i]])
            layer_loss = Lambda(lambda x:x[0]*layer_weight/x[1])([layer_loss, layer_shape])
            losses.append(layer_loss)
    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) * args["style_w"])(losses)

    loss_model = Model(model.inputs + targets, loss)

    return loss_model, model, targets
