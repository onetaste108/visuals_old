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
            "style_chain": False,
            "style_w": 1,
            "style_layers": [1,4,7,11,15],
            "style_lw": [1,1,1,1,1],
            "style_scale": 1,
            "style_offset": -20,

            "content": True,
            "content_w": 1,
            "content_layers": [12],

            "variational": True,
            "variational_w": 1,
            "variational_pow": 1.25,

            "octaves": 1,
            "octave_a": .4,

            "size": [512,512],
            "iters": 10,
            "iter": 20,

            "x0": "content",

            "model": "vgg16",
            "pool": "avg",
            "padding": "valid"
        }
        self.model = None
        self.eval = None
        self.feed = {}
        self.modules = {}
        self.tgs = {}

    def build(self):
        self.model, self.modules = build(self.args)
        self.eval = self.make_eval()



    def render(self):
        if self.args["x0"] == "content":
            x0 = self.feed["content"]
        elif self.args["x0"] == "noise":
            x0 = np.random.randn(self.args["size"][1],self.args["size"][0],3)
        else:
            x0 = preprocess(imsize(self.args["x0"], self.args["size"]))
        callback = self.make_callback()
        bounds = get_bounds(x0)
        print("Render begins")
        for i in range(self.args["iters"]):
            x0, min_val, info = fmin_l_bfgs_b(callback, x0.flatten(), bounds=bounds, maxfun=self.args["iter"])

            imshow(deprocess(x0.reshape((self.args["size"][1], self.args["size"][0], 3))))
            print("Iteration",i,"of",self.args["iters"])
        return deprocess(x0.reshape((self.args["size"][1], self.args["size"][0], 3)))


    def make_eval(self):
        grads = K.gradients(self.model.output, self.model.inputs[0])[0]
        outputs = [self.model.output] + [grads]
        return K.function(self.model.inputs, outputs)

    def make_callback(self):
        def fn(x):
            x = x.reshape((1, self.args["size"][1], self.args["size"][0], 3))
            outs = self.eval([x]+self.tgs)
            loss_value = outs[0]
            grad_values = np.array(outs[1:]).flatten().astype('float64')
            return loss_value, grad_values
        return fn



    def setup(self):
        tgs = []
        for m in ["content", "style", "style_chain"]:
            if self.modules[m] is not None:
                t = self.modules[m].predict(np.array([self.feed[m]]))
                if type(t) is list: tgs.extend(t)
                else: tgs.append(t)
        self.tgs = tgs

    def set_module(self,module,img):
        if module == "style":
            img = preprocess(imsize(img, factor = impropscale(img.shape[:2],self.args["size"][::-1]) * self.args["style_scale"]))
            self.feed["style"] = img
            self.feed["style_chain"] = img
        elif module == "content":
            img = preprocess(imsize(img, self.args["size"]))
            self.feed[module] = img

    def clear(self):
        for k in self.tgs: del self.tgs[k]
        for k in self.feed: del self.feed[k]
        self.modules = None
        self.model = None
        K.clear_session()
