from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from netty import model_vgg
from netty import model_variational
from netty import module_content
from netty import module_style



def build(args):
    input = Input((None,None,3))

    losses = []
    module_inputs = []
    modules = {}

    if args["variational"]:
        loss_model = model_variational.build(args)
        losses.append(loss_model(input))

    if args["content"]:
        loss_model, target_model, targets = module_content.build(args)
        losses.append(loss_model([input] + targets))
        module_inputs.extend(targets)
        modules["content"] = target_model
    else: modules["content"] = None

    if args["style"]:
        loss_model, target_model, targets = module_style.build(args)
        losses.append(loss_model([input] + targets))
        module_inputs.extend(targets)
        modules["style"] = target_model
    else: modules["style"] = None

    if len(losses) > 0:
        loss = Lambda(lambda x: K.expand_dims(K.sum(x) / len(losses)))(losses)
    else:
        print("Nothing to optimize")
        return

    model = Model([input] + module_inputs, loss)
    return model, modules
