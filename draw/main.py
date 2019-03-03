import pyglet
from draw import Draw
import numpy as np

app = pyglet.window.Window(1024,1024,resizable = True)
draw = Draw()
sd2d = draw.sd2d

time = 0
MOD_1 = False
MOD_2 = False
from pyglet.window import key

mat = np.float32([[1,0],
                  [0,1]])
mat0 = np.copy(mat)

sd2d.viewport(2,2)

@app.event
def on_draw():
    global time
    global MOD_1
    global MOD_2
    draw.clear()

    sd2d.color([1,0,1,0.5])
    sd2d.stroke()
    sd2d.stroke_weight(0.001)
    sd2d.smooth(0.002)

    for y in range(21):
        sd2d.line(-1,y/20*2-1,1,y/20*2-1)
    for x in range(21):
        sd2d.line(x/20*2-1,-1,x/20*2-1,1)

    sd2d.fill()
    sd2d.color([0,1,1,1])
    sd2d.circle(0,0.2,0.01)
    sd2d.circle(0,-0.2,0.01)
    sd2d.circle(0.2,-0.2,0.01)
    sd2d.circle(-0.2,-0.2,0.01)
    sd2d.circle(0.2,0,0.01)
    sd2d.circle(-0.2,0,0.01)

    sd2d.stroke()
    sd2d.stroke_weight(0.002)
    sd2d.color([1,0,0,1])
    sd2d.line(0,0,mat[0][0],mat[0][1])

    sd2d.stroke()
    sd2d.stroke_weight(0.002)
    sd2d.color([0,0,1,1])
    sd2d.line(0,0,mat[1][0],mat[1][1])

    sd2d.fill()
    sd2d.color([0,1,0,1])
    sd2d.circle(0,0,0.01)

@app.event
def on_mouse_motion(x,y,dx,dy):
    global MOD_1
    global MOD_2

    # if MOD_1: mat[0] += np.float32([dx/app.width*2,dy/app.height*2])
    # if MOD_2: mat[1] += np.float32([dx/app.width*2,dy/app.height*2])



@app.event
def on_key_press(symbol, modifiers):
    global MOD_1
    global MOD_2

    if symbol == key._1: MOD_1 = True
    if symbol == key._2: MOD_2 = True
    pass

@app.event
def on_key_release(symbol, modifiers):
    global MOD_1
    global MOD_2

    if symbol == key._1: MOD_1 = False
    if symbol == key._2: MOD_2 = False
    pass

@app.event
def on_resize(w,h):
    pass

def update(dt):
    global time
    time += dt
    pass

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
