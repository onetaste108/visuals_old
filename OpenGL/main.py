import pyglet
from draw import Draw
import numpy as np

class App(pyglet.window.Window):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.d = Draw()
        self.mousex = 0
        self.mousey = 0
        self.time = 0

    def on_draw(self):
        self.d.ctx.clear()
        self.d.sd2d_circle(self.mousex/self.width*2-1,
                        self.mousey/self.height*2-1,
                        np.sin(self.time)/2+0.5)

    def update(self,dt):
        self.time += dt
        pass

    def on_mouse_motion(self,x,y,dx,dy):
        self.mousex = x
        self.mousey = y

    def run(self):
        pyglet.clock.schedule(self.update)
        pyglet.app.run()

app = App(512,512)
app.run()
