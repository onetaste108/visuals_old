import keras
from keras.applications import vgg19
from keras.applications import vgg16
import os

import numpy as np
from keras.layers import Lambda, Concatenate, Input
from keras.models import Model
from keras import backend as K
from netty.vgg_utils import get_vgg_shape

def create_model(input_size, model="vgg19", pool="avg", padding="same", patch=256):
    if model == "vgg19": default_model = vgg19.VGG19(weights="imagenet", include_top=False)
    elif model == "vgg16": default_model = vgg16.VGG16(weights="imagenet", include_top=False)
    outs = []
    for i, layer in enumerate(default_model.layers):
        if i == 0:
            input = Input((input_size[0],input_size[1],3))
            output = input
            outs.append(output)
        else:
            if isinstance(layer, keras.layers.Conv2D):
                old_layer_shape = get_vgg_shape(input_size, i-1, octave=0, model=model, padding=padding)
                config = layer.get_config()
                config["padding"] = padding
                conv = keras.layers.Conv2D(config["filters"],
                    config["kernel_size"],
                    padding=config["padding"],
                    data_format=config["data_format"],
                    activation=config["activation"],
                    weights=layer.get_weights()
                    )
                output = build_patch_model(output,patch,conv,old_layer_shape)
                outs.append(output)
            elif isinstance(layer, keras.layers.MaxPooling2D):
                config = layer.get_config()
                config["padding"] = padding
                if pool=="avg": output = keras.layers.AveragePooling2D.from_config(config)(output)
                else: output = keras.layers.MaxPooling2D.from_config(config)(output)
                outs.append(output)
    m = Model(inputs=input,outputs=outs)
    m.summary()
    return m

def build_patch_model(input,patch,conv,old_layer_shape):

    y_n = (old_layer_shape[0]-patch)//(patch-2)+1
    x_n = (old_layer_shape[1]-patch)//(patch-2)+1
    y_rem = old_layer_shape[0] - ((y_n-1)*(patch-2)+patch) +2
    x_rem = old_layer_shape[1] - ((x_n-1)*(patch-2)+patch) +2
    y_sh = y_n
    x_sh = x_n
    if y_rem >= 3: y_sh += 1
    if x_rem >= 3: x_sh += 1
    p_s = np.empty([y_sh,x_sh,2,2], dtype=np.int32)
    for y in range(y_sh):
        for x in range(x_sh):
            if y < y_n and x < x_n:
                p_s[y][x] = np.int32([[y*(patch-2),y*(patch-2)+patch],[x*(patch-2),x*(patch-2)+patch]])
            elif y == y_n and x < x_n:
                p_s[y][x] = np.int32([[old_layer_shape[0]-y_rem,old_layer_shape[0]],[x*(patch-2),x*(patch-2)+patch]])
            elif y < y_n and x == x_n:
                p_s[y][x] = np.int32([[y*(patch-2),y*(patch-2)+patch],[old_layer_shape[0]-y_rem,old_layer_shape[1]]])
            else:
                p_s[y][x] = np.int32([[old_layer_shape[0]-y_rem,old_layer_shape[0]],[old_layer_shape[0]-y_rem,old_layer_shape[1]]])

    conv_layers = []
    for y in range(y_sh):
        conv_layers.append([])
        for x in range(x_sh):
            splited = split(p_s[y][x])(input)
            conved = conv(splited)
            conv_layers[-1].append(conved)

    if len(conv_layers) > 1:
        x_conc = []
        for y in range(y_sh):
            conced = Concatenate(2)(conv_layers[y])
            x_conc.append(conced)
    else:
        x_conc = conv_layers[0]
    if len(x_conc) > 1:
        out = Concatenate(1)(x_conc)
    else:
        out = x_conc[0]

    return out

def split(s):
    def fn(x):
        x = x[0]
        x = x[s[0][0]:s[0][1],s[1][0]:s[1][1]]
        return K.expand_dims(x,axis=0)
    return Lambda(fn)

def build(args):
    return create_model(args["size"],args["model"],args["pool"],args["padding"],args["patch_window"])
