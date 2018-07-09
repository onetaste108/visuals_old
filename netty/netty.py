from netty.build import build
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize

from keras import backend as K
from img_utils import *
from netty.vgg_utils import *

class Netty:
    def __init__(self):
        self.args = {
            "style": True,
            "style_w": 1,
            "style_layers": [3,6],

            "content": True,
            "content_w": 1,
            "content_layers": [6,7],

            "octaves": 1,
            "octave_a": .4,

            "size": [512,512],
            "iters": 10,

            "model": "vgg16",
            "pool": "avg",
            "padding": "valid"
        }
        self.feed = {
            "x0": None
        }
        self.model = None
        self.modules = None
        self.eval = None

    def build(self):
        self.model, self.modules = build(self.args)
        self.eval = self.make_eval()

    def set_tgs(self, img):
        tgs = []
        for i in range(len(self.modules)):
            if i == 1: self.feed["content"] = img[i]
            t = self.modules[i].predict(np.array([preprocess(img[i])]))
            tgs.extend(t)
        self.tgs = tgs

    def set_x0(self,img=None):
        if img == "content":
            self.feed["x0"] = self.feed["content"]

    def render(self):
        if self.feed["x0"] is None:
            x0 = np.random.randn(self.args["size"][0],self.args["size"][1],3).flatten()
        else:
            x0 = self.feed["x0"]
        callback = self.make_callback()
        bounds = get_bounds(x0)
        print("Render begins")
        for i in range(self.args["iters"]):
            x0, min_val, info = fmin_l_bfgs_b(callback, x0.flatten(), bounds=bounds, maxfun=20)

            imshow(deprocess(x0.reshape((self.args["size"][1], self.args["size"][1], 3))))
            print("Iteration",i,"of",self.args["iters"])
        return deprocess(x0.reshape((self.args["size"][1], self.args["size"][1], 3)))

    def make_eval(self):
        grads = K.gradients(self.model.output, self.model.inputs[0])[0]
        outputs = [self.model.output] + [grads]
        return K.function(self.model.inputs, outputs)

    def make_callback(self):
        def fn(x):
            x = x.reshape((1, self.args["size"][1], self.args["size"][1], 3))
            outs = self.eval([x]+self.tgs)
            loss_value = outs[0]
            grad_values = np.array(outs[1:]).flatten().astype('float64')
            return loss_value, grad_values
        return fn


class Evaluator():
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        self.grads = None
        self.fn = None

    def compute(self, model):
        grads = K.gradients(model.output, model.inputs[0])
        if len(grads) == 1: grads = grads[0]
        else: grads = K.concatenate(grads, axis=0)
        outputs = [model.output] + [grads]
        print(outputs)
        return K.function(model.inputs, outputs)

    def loss(self, x, size, tgs):
        assert self.loss_value is None
        loss_value, grad_values = self.eval(x,size,tgs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def eval(self, x, size, targets):
        x = x.reshape((1, size[1], size[0], 3))
        outs = self.fn([x]+targets[0])
        loss_value = outs[0]
        grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
