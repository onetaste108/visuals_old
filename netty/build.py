import keras
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from keras import backend as K

from netty import model_vgg
from netty import module_style
from netty import module_content

def build(args):
    input = Input((None,None,3))

    losses = []
    module_inputs = []
    module_models = []

    if args["style"]:
        loss_model, target_model, targets = module_style.build(args)
        losses.append(loss_model([input] + targets))
        module_models.append(target_model)
        module_inputs.extend(targets)

    if args["content"]:
        loss_model, target_model, targets = module_content.build(args)
        losses.append(loss_model([input] + targets))
        module_models.append(target_model)
        module_inputs.extend(targets)

    if len(losses) > 0:
        loss = Lambda(lambda x: K.expand_dims(K.sum(x) / len(losses)))(losses)
    else:
        print("Nothing to optimize")
        return

    model = Model([input] + module_inputs, loss)
    return model, module_models
