from netty.netty import Netty
import img_utils as im
import numpy as np
from netty import netty_utils as nutil
from IPython.display import clear_output


def image(path, s = 1):
    global images
    global scales
    images.append(im.load(path))
    scales.append(s)
    
def background(path = None):
    global x0
    if path is None:
        x0 = None
        return
    x0 = im.load(path)
    
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
    its = 300
    
def reset():
    global la
    global x0
    global res
    global ires
    global st
    global its
    global images
    global scales
    images = []
    scales = []
    
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
        net.set_style(images, None, scales, None)
        net.set_x0(x)
        print("Setup...")
        net.setup()
        print("Render!")
        x=net.render()
        clear_output()
        im.save_frame(x,"data/bin",q=95)