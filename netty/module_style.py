from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from keras.layers import Input, Lambda
from keras import backend as K

def gram_l():
    def fn(x):
        shape = K.shape(x)
        x = K.reshape(x, (-1, shape[3]))
        x = K.dot(K.transpose(x),x)
        return K.expand_dims(x, axis=0)
    return Lambda(fn)

def build(args):
    vgg = model_vgg.build(args)
    vgg = extract_layers(vgg, args["style_layers"])
    gram_layer = gram_l()
    gram_model = attach_models(vgg, gram_layer)
    octave_model = model_octave.build(args["octaves"], args["octave_a"])
    model = attach_models(octave_model, gram_model)

    targets = []
    losses = []
    for i in range(len(model.outputs)):
        targets.append(Input(model.outputs[i].shape[1:]))
        layer_weight = 1
        layer_loss = Lambda(lambda x: K.expand_dims(K.sum(K.square(x[0]-x[1]))) / K.cast(K.shape(x), "float32") * layer_weight)([targets[i], model.outputs[i]])
        losses.append(layer_loss)
    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) / len(losses) * args["style_w"])(losses)

    loss_model = Model(model.inputs + targets, loss)

    return loss_model, model, targets
