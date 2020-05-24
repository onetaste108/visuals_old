from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model

def variation_l(weight=0.01, power=1.25):
    def fn(x):
        shape = K.shape(x)
        a = K.square(x[:,1:,:shape[2]-1,:] - x[:,:shape[1]-1,:shape[2]-1,:])
        b = K.square(x[:,:shape[1]-1,1:,:] - x[:,:shape[1]-1,:shape[2]-1,:])
        return K.expand_dims(K.sum(K.pow(a+b, power)) * weight)
    return Lambda(fn)

def build(args):
    inputs = Input(shape=(None,None,3))
    outs = variation_l(args["variational_w"],args["variational_pow"])(inputs)
    return Model(inputs=inputs, outputs=outs)
