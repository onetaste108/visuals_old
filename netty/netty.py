from netty.build import build
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import backend as K
import img_utils as im
from netty.vgg_utils import *
from netty import gram_patcher
from netty import netty_utils as nutil
tf.compat.v1.disable_eager_execution()
class Netty:
    def __init__(self):
        self.args = {
            "style": True,
            "style_chain": False,
            "style_w": 1,
            "style_layers": [1,4,7,12,17],
            "style_lw": [1,1,1,1,1],

            "content": False,
            "content_w": 1,
            "content_layers": [12],

            "variational": False,
            "variational_w": 1e-1,
            "variational_pow": 1.25,

            "size": [512,512],

            "patch_window": 1024*2,
            "patch_window_ref_stride": 64,
            "patch_overlay": 64,
            "patch_maxfun": 10,
            "patch_iters": 1,
            "patch_display": True,

            "maxfun": 100,
            "display": 25,
            "callback_fn": None,

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

    def make_eval(self):
        grads = K.gradients(self.model.output, self.model.inputs[0])[0]
        outputs = [self.model.output] + [grads]
        return K.function(self.model.inputs, outputs)

    def make_callback(self,shape,tgs):
        i = [0]
        display = self.args["display"]
        cbfn = self.args["callback_fn"]
        def fn(x):
            x = x.reshape((1, shape[0], shape[1], 3))
            outs = self.eval([x]+tgs)
            loss_value = outs[0]
            grad_values = np.array(outs[1:]).flatten().astype('float64')

            if cbfn is not None:
                cbfn(i[0],deprocess(x[0]))

            if i[0] % 10 == 0 and i[0] != 0:
                print(i[0],end=" ")
            else:
                print(".",end=" ")

            if i[0] % display == 0 and i[0] != 0:
                im.show(deprocess(x[0]))
            i[0] += 1

            return loss_value, grad_values
        return fn

    def render(self):
        x0 = self.feed["x0"]
        x0m = self.feed["x0_mask"]

        x = self.render_patch(x0,self.tgs,self.args["maxfun"])

        x = deprocess(x)
        im.show(x)
        print("Rendered!")
        return x

    def render_patched(self):
        x0 = self.feed["x0"].copy()
        x0m = self.feed["x0_mask"]

        p_iters = self.args["patch_iters"]
        p_maxfun = self.args["patch_maxfun"]

        for it in range(p_iters):
            patches, tgs = gram_patcher.match(x0,self.feed["x0_mask"],self.feed["style"],self.feed["style_masks"],self.modules["style"],self.args["patch_window"])
            m,n = patches.shape[:2]
            for i in range(m):
                for j in range(n):
                    x0_patch = x0[patches[i][j][0][0]:patches[i][j][0][1],patches[i][j][1][0]:patches[i][j][1][1]]
                    x0m_patch = x0m[patches[i][j][0][0]:patches[i][j][0][1],patches[i][j][1][0]:patches[i][j][1][1]]
                    tg = nutil.mix_tgs(tgs[i][j])

                    patch = self.render_patch(patch,x0m_patch+tg,p_maxfun)
                    x0[patches[i][j][0][0]:patches[i][j][0][1],patches[i][j][1][0]:patches[i][j][1][1]] = patch
                    print("Patch",i,j,"of",m,n)
                    if self.args["patch_display"]:
                        im.show(deprocess(patch))

            im.show(deprocess(x0))

        return deprocess(x0.reshape((self.args["size"][1], self.args["size"][0], 3)))

    def render_patch(self,x0,tgs,maxfun=1):
        shape = x0.shape
        bounds = get_bounds(x0)
        callback = self.make_callback(shape=shape[:2],tgs=tgs)
        x, min_val, info = fmin_l_bfgs_b(callback, x0.flatten(), bounds=bounds, maxfun=maxfun)
        return x.reshape(shape)

    def setup(self):

        tgs = []
        if self.args["content"]:
            t = self.get_content_tgs()
            tgs.append(t)
        if self.args["style"]:
            mask = self.feed["x0_mask"]
            if type(mask) is list: tgs.extend(mask)
            else: tgs.append(mask)
                
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
        return nutil.mix_tgs(tgs,self.args["style_imgs_w"])

    def get_content_tgs(self):
        tgs = self.modules["content"].predict(np.array([self.feed["content"]]))
        return tgs

    def set_tgs(self,tgs):
        self.tgs = tgs

    def set_style(self,imgs,masks=None,scales=None,w=None):
        self.args["style_imgs_w"] = w
        if type(imgs) is not list: imgs = [imgs]

        if masks is None:
            masks = [None for i in range(len(imgs))]
        else:
            if type(masks) is not list: masks = [masks]

        if scales is None:
            scales = [1 for i in range(len(imgs))]
        else:
            if type(scales) is not list:
                scales = [scales for i in range(len(imgs))]

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

    def set_x0(self,img=None,mask=None):
        if img is None:
            self.feed["x0"] = np.random.randn(self.args["size"][1],self.args["size"][0],3) * 10
        else:
            img = preprocess(im.size(img, self.args["size"]))
            self.feed["x0"] = img

        if mask is not None:
            mask = im.size(mask,self.args["size"])
            l_mask = scale_mask(mask,self.args["style_layers"])
        else:
            l_mask = []
            for l in self.args["style_layers"]:
                vgg_shape = get_vgg_shape(self.args["size"],l)[:-1]
                l_mask.append(np.ones([1,vgg_shape[1],vgg_shape[0]],np.float32))
        self.feed["x0_mask"] = l_mask

    def clear(self):
        K.clear_session()
