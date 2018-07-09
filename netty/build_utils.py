from keras.models import Model

def extract_layers(base_model, layers):
    model_layers = []
    inputs = base_model.inputs
    for l in layers:
        model_layers.append(base_model.layers[l].output)
    return Model(inputs, model_layers)

def attach_models(base_model, model):
    inputs = base_model.inputs
    outs = []
    for o in base_model.outputs:
        out = model(o)
        if type(out) == list:
            outs.extend(out)
        else:
            outs.append(out)
    return Model(inputs, outs)
