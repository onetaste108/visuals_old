from tensorflow.keras.models import Model

def extract_layers(base_model, layers):
    inputs = base_model.input
    model_layers = []
    for l in layers:
        model_layers.append(base_model.layers[l].output)
    return Model(inputs, model_layers)

def extract_outputs(base_model, layers):
    inputs = base_model.input
    outs = []
    for o in layers:
        outs.append(base_model.outputs[o])
    return Model(inputs, outs)

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
