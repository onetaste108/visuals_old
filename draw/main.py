import pyglet
from draw import Draw
from anima import Anima
import numpy as np

app = pyglet.window.Window(600,600,resizable = True)
draw = Draw()
anima = Anima()
sd2d = draw.sd2d

time = 0
MOD_1 = False
MOD_2 = False
MOD_3 = False
from pyglet.window import key
sd2d.viewport(10,10)

ih = np.float32([1,0])
jh = np.float32([0,1])



@app.event
def on_draw():
    global time
    global MOD_1
    global MOD_2
    draw.clear()

    sd2d.color([0.2,0.1,0.3,1])
    sd2d.stroke()
    sd2d.stroke_weight(0.02)
    sd2d.smooth(0.01)

    for y in range(21):
        if y % 2 == 0: sd2d.stroke_weight(0.05)
        else: sd2d.stroke_weight(0.01)
        sd2d.line(-5,y/20*10-5,5,y/20*10-5)
    for x in range(21):
        if x % 2 == 0: sd2d.stroke_weight(0.05)
        else: sd2d.stroke_weight(0.01)
        sd2d.line(x/20*10-5,-5,x/20*10-5,5)

    sd2d.fill()
    sd2d.color([0,1,1,1])

    #ih
    sd2d.stroke()
    sd2d.stroke_weight(0.05)
    sd2d.color([1,0,0,1])
    sd2d.line(0,0,1,0)
    #jh
    sd2d.stroke()
    sd2d.stroke_weight(0.05)
    sd2d.color([0,0,1,1])
    sd2d.line(0,0,0,1)

    sd2d.fill()
    sd2d.color([0,1,0,1])
    sd2d.circle(0,0,0.1)

@app.event
def on_mouse_motion(x,y,dx,dy):
    global MOD_1
    global MOD_2
    global MOD_2
    global ih, jh

    if MOD_1: ih += np.float32([dx/app.width*10,dy/app.height*10])
    if MOD_2: jh += np.float32([dx/app.width*10,dy/app.height*10])
    if MOD_3:
        theta = dx/app.width*np.pi*4
        tih = np.float32([np.cos(theta), np.sin(theta)])
        tjh = np.float32([-np.sin(theta), np.cos(theta)])
        ih = ih[0]*tih + ih[1]*tjh
        jh = jh[0]*tih + jh[1]*tjh
        pass



@app.event
def on_key_press(symbol, modifiers):
    global MOD_1
    global MOD_2
    global MOD_3

    if symbol == key._1: MOD_1 = True
    if symbol == key._2: MOD_2 = True
    if symbol == key._3: MOD_3 = True
    pass

@app.event
def on_key_release(symbol, modifiers):
    global MOD_1
    global MOD_2
    global MOD_3

    if symbol == key._1: MOD_1 = False
    if symbol == key._2: MOD_2 = False
    if symbol == key._3: MOD_3 = False
    pass

@app.event
def on_resize(w,h):
    pass

def update(dt):
    global time
    time += dt
    sd2d.prog["ih"].value = tuple(ih)
    sd2d.prog["jh"].value = tuple(jh)
    pass

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
