from netty.netty import Netty
import img_utils as im
import numpy as np
from netty import netty_utils as nutil
from IPython.display import clear_output
import os

def load_folder(path):
    files = os.listdir(path)
    imgs = []
    for f in files:
        try:
            imgs.append(im.load(os.path.join(path,f)))
        except:
            print("Failed to load",f)
    return imgs

def image(img, s = 1, strech = True, mask=None):
    global images
    global scales
    global masks
    images.append(img)
    masks.append(mask)
    scales.append([s,img.shape[:2][::-1],strech])
    
def images(*args):
    for a in args:
        image(*a)
    
def background(img=None):
    global x0
    if img is None:
        x0 = None
        return
    x0 = img
    
def layers(l = [1,4,7]):
    global la
    la = l
    
def resolution(r = [512, 512]):
    global res
    res = r
    
def from_resolution(r = 512//2):
    global ires
    ires = r
    
def step(s = 1.2):
    global st
    st = 1.2
    
def iters(it = 300):
    global its
    its = it
    
def reset():
    global la
    global x0
    global res
    global ires
    global st
    global its
    global images
    global scales
    global masks
    images = []
    scales = []
    masks = []
    
    layers()
    resolution()
    from_resolution()
    step()
    iters()
    
def run():
    global la
    global x0
    global res
    global ires
    global st
    global its
    global images
    global scales
    
    _scales = scales
    scales = []
    for s in _scales:
        if s[2]: scales.append(s[0])
        else: scales.append(im.propscale(np.float32(res), np.float32(s[1]))*s[0])
    
    net = Netty()
    net.clear()
    net.args["style_layers"] = la
    net.build()
    
    x = x0
    s,m = nutil.incremental(res,int(ires),st,its)

    for s_,m_ in zip(s,m):
        print(s_)
        print(m_)
        net.args["size"] = s_
        net.args["maxfun"] = m_
        print("Setting style...")
        net.set_style(images, masks, scales, None)
        net.set_x0(x)
        print("Setup...")
        net.setup()
        print("Render!")
        x=net.render()
        clear_output()
        im.save_frame(x,"data/bin",q=95)