import keras
from keras.applications import vgg19
from keras.applications import vgg16
import os

def create_model(model="vgg19", pool="avg", padding="same"):
    if model == "vgg19": default_model = vgg19.VGG19(weights="imagenet", include_top=False)
    elif model == "vgg16": default_model = vgg16.VGG16(weights="imagenet", include_top=False)
    new_layers = []
    for i, layer in enumerate(default_model.layers):
        if i == 0: new_layers.append(layer)
        else:
            if isinstance(layer, keras.layers.Conv2D):
                config = layer.get_config()
                config["padding"] = padding
                new_layers.append(keras.layers.Conv2D.from_config(config))
            elif isinstance(layer, keras.layers.MaxPooling2D):
                config = layer.get_config()
                config["padding"] = padding
                if pool=="avg": new_layers.append(keras.layers.AveragePooling2D.from_config(config))
                else: new_layers.append(keras.layers.MaxPooling2D.from_config(config))
    model = keras.models.Sequential(layers=new_layers)
    for new, old in zip(model.layers, default_model.layers): new.set_weights(old.get_weights())
    return model

def load_model(model="vgg19", pool="avg", padding="same"):
    path = os.path.join("models",model+"_"+pool+"_"+padding+".h5")
    if not os.path.exists(path):
        print("VGG model not found. Creating model...")
        if not os.path.exists("models"): os.mkdir("models")
        m = create_model(model=model,pool=pool,padding=padding)
        m.save(path)
    else: m = keras.models.load_model(path)
    return m

def build(args):
    return load_model(args["model"],args["pool"],args["padding"])
