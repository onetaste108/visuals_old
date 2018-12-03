from netty.build import build
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize

from keras import backend as K
import img_utils as im
from netty.vgg_utils import *
from netty import gram_patcher
from netty import netty_utils as nutil

class Netty:
    def __init__(self):
        self.args = {
            "style": True,
            "style_chain": False,
            "style_w": 1,
            "style_layers": [1,4,7,12,17],
            "style_lw": [1,1,1,1,1],
            "style_scale": 1,
            "style_offset": 0,

            "content": False,
            "content_w": 1,
            "content_layers": [12],

            "variational": True,
            "variational_w": 1e-1,
            "variational_pow": 1.25,

            "octaves": 1,
            "octave_a": .4,

            "size": [512,512],

            "patch_window": 1024*2,
            "patch_window_ref_stride": 64,
            "patch_overlay": 64,
            "patch_maxfun": 10,
            "patch_iters": 1,
            "patch_display": True,

            "maxfun": 100,
            "disp_int": 50,

            "x0": "noise",

            "model": "vgg19",
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
            x0 = np.random.randn(self.args["size"][1],self.args["size"][0],3) * 10
        else:
            x0 = preprocess(im.size(self.args["x0"], self.args["size"]))

        if min(self.args["size"][0],self.args["size"][1]) > self.args["patch_window"]:
            print("Rendering patch...")
            x = self.render_patched(x0,self.args["patch_iters"], self.args["patch_maxfun"])
        else:
            # self.setup()
            print("Rendering one...")
            x = self.render_one(x0)
        print("Rendered!")
        im.show(x)
        return x

    def render_one(self,x0):

        callback = self.make_callback()
        bounds = get_bounds(x0)

        x0, min_val, info = fmin_l_bfgs_b(callback, x0.flatten(), bounds=bounds, maxfun=self.args["maxfun"])

        return deprocess(x0.reshape((self.args["size"][1], self.args["size"][0], 3)))

    def render_patched(self,x0,iters=1,maxfun=10):
        if self.args["x0"] == "content":
            x0 = self.feed["content"]
        elif self.args["x0"] == "noise":
            x0 = np.random.randn(self.args["size"][1],self.args["size"][0],3) * 10
        else:
            x0 = preprocess(im.size(self.args["x0"], self.args["size"]))

        patches, tgs = gram_patcher.match(x0,self.feed["style"],self.modules["style"],self.args["patch_window"])
        m,n = patches.shape[:2]
        for it in range(iters):
            for i in range(m):
                for j in range(n):
                    patch = x0[patches[i][j][0][0]:patches[i][j][0][1],patches[i][j][1][0]:patches[i][j][1][1]]
                    patch = self.render_patch(patch,tgs[i][j],maxfun)
                    x0[patches[i][j][0][0]:patches[i][j][0][1],patches[i][j][1][0]:patches[i][j][1][1]] = patch
                    print("Patch",i,j,"of",m,n)
                    if self.args["patch_display"]:
                        im.show(deprocess(patch))

        return deprocess(x0.reshape((self.args["size"][1], self.args["size"][0], 3)))

    def render_patch(self,x0,tgs,iters=1):
        shape = x0.shape
        bounds = get_bounds(x0)
        callback = self.make_callback(shape=shape[:2],tgs=tgs)
        x0, min_val, info = fmin_l_bfgs_b(callback, x0.flatten(), bounds=bounds, maxfun=iters)
        return x0.reshape(shape)

    def make_eval(self):
        grads = K.gradients(self.model.output, self.model.inputs[0])[0]
        outputs = [self.model.output] + [grads]
        return K.function(self.model.inputs, outputs)

    def make_callback(self,shape=None,tgs=None):
        i = [0]
        disp_int = self.args["disp_int"]
        if shape is None:
            shape = self.args["size"][::-1]
        if tgs is None:
            tgs = self.tgs
        def fn(x):
            x = x.reshape((1, shape[0], shape[1], 3))
            outs = self.eval([x]+tgs)
            loss_value = outs[0]
            grad_values = np.array(outs[1:]).flatten().astype('float64')
            print(".",end=" ")
            if i[0] % disp_int == 0 and i[0] != 0:
                im.show(deprocess(x[0]))
            i[0] += 1

            return loss_value, grad_values
        return fn

    def setup(self):
        tgs = []
        for m in ["content", "style"]:
            if self.args[m]:
                if m == "content":
                    t = self.get_content_tgs()
                elif m == "style":
                    t = self.get_style_tgs()
                if type(t) is list: tgs.extend(t)
                else: tgs.append(t)
        self.set_tgs(tgs)

    def get_style_tgs(self):
        n = len(self.feed["style"])
        tgs = []
        for i in range(n):
            t = self.modules["style"].predict([np.array([self.feed["style"][i]])]+self.feed["style_masks"][i])
            tgs.append(t)
        return nutil.mix_tgs(tgs)

    def get_content_tgs(self):
        tgs = self.modules["content"].predict(np.array([self.feed["content"]]))
        return tgs

    def set_tgs(self,tgs):
        self.tgs = tgs

    def set_style(self,imgs,masks=None,scales=None):
        if type(imgs) is not list: imgs = [imgs]
        if masks is None:
            masks = [None for i in range(len(imgs))]
        if scales is None:
            scales = [1 for i in range(len(imgs))]
        else:
            if type(scales) is not list: scales = [scales]

        self.feed["style"] = []
        self.feed["style_masks"] = []

        for img, mask, scale in zip(imgs,masks,scales):
            if scale == 0:
                factor = 1
            else:
                factor=im.propscale(img.shape[:2],self.args["size"][::-1]) * scale
            img = preprocess(im.size(img, factor=factor))
            self.feed["style"].append(img)

            if mask is not None:
                mask = im.size(mask,img.shape[:2][::-1])
                l_mask = scale_mask(mask,self.args["style_layers"])
            else:
                l_mask = []
                for l in self.args["style_layers"]:
                    vgg_shape = get_vgg_shape(img.shape[:2],l)[:-1]
                    l_mask.append(np.ones([1,vgg_shape[0],vgg_shape[1]],np.float32))
            self.feed["style_masks"].append(l_mask)

    def set_content(self,img):
        img = preprocess(im.size(img, self.args["size"]))
        self.feed["content"] = img

    def set_module(self,module,img,mask=None):
        if module == "style":
            img = preprocess(im.size(img, factor = im.propscale(img.shape[:2],self.args["size"][::-1]) * self.args["style_scale"]))
            self.feed["style"] = img
            self.args["style_shape"] = img.shape[:2]
            if mask is not None:
                mask = im.size(mask,img.shape[:2][::-1])
                masks = scale_mask(mask,self.args["style_layers"])

            else:
                masks = []
                for l in self.args["style_layers"]:
                    masks.append(np.ones(get_vgg_shape(img.shape[:2],l)[:-1],np.float32))
            for i in range(len(masks)):
                masks[i] = np.array([masks[i]])
            self.feed["style_masks"] = masks
        elif module == "content":
            img = preprocess(im.size(img, self.args["size"]))
            self.feed[module] = img

    def clear(self):
        for k in self.tgs: del self.tgs[k]
        for k in self.feed: del self.feed[k]
        self.modules = None
        self.model = None
        K.clear_session()
