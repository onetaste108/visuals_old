from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from tensorflow.keras.layers import Input, Lambda, Multiply, Concatenate
from tensorflow.keras import backend as K
import tensorflow as tf

def apply_mask():
    def fn(x):
        x1 = x[0][0]
        x2 = x[1][0]
        x2 = K.expand_dims(x2,-1)
        x1 = tf.multiply(x1,x2)
        return K.expand_dims(x1,axis=0)
    return Lambda(fn)

def mask_gram_l():
    def fn(x):
        mask = x[1][0]
        mask = K.cast(tf.math.count_nonzero(mask),"float32")
        x = x[0]
        shape = K.shape(x)
        x = K.reshape(x, (-1, shape[3]))
        x = K.dot(K.transpose(x),x)
        shape = K.cast(shape[1:],"float32")
        x = x / mask / shape[-1]
        return K.expand_dims(x, axis=0)
    return Lambda(fn)

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

def build_mask_model(model):
    apmask = apply_mask()
    mask_gram_layer = mask_gram_l()
    mask_inputs = []
    mask_outs = []
    for i,o in enumerate(model.outputs):
        mask_input = Input((o.shape[1],o.shape[2]),name="mask_input_"+str(i))
        mask_inputs.append(mask_input)
        mask_output = apmask([o,mask_input])
        mask_output = mask_gram_layer([mask_output,mask_input])
        mask_outs.append(mask_output)
    mask_model = Model(model.inputs+mask_inputs,mask_outs)
    return mask_model, mask_inputs


def build(args):
    vgg = model_vgg.build(args)
    model = extract_layers(vgg, args["style_layers"])
    mask_model, mask_inputs = build_mask_model(model)

    targets = []
    losses = []
    for i in range(len(mask_model.outputs)):
        targets.append(Input(mask_model.outputs[i].shape[1:],name="wtf"+str(i)))
        layer_weight = args["style_lw"][i] / len(args["style_layers"])
        layer_loss = loss_l(layer_weight)([targets[i], mask_model.outputs[i]])
        losses.append(layer_loss)
    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) * args["style_w"])(losses)

    loss_model = Model(mask_model.inputs + targets, loss)
    return loss_model, mask_model, mask_inputs+targets
