import pyglet
from pyglet.gl import *
import math

class Draw:

    def __init__(self):
        self.win = pyglet.window.Window(100,100)

    # def draw(self):
    #     pass
    #
    # def run(self):
    #     @win.event
    #     def draw():
    #         self.win.clear()
    #         self.draw()
    #
    #     pyglet.app.run()
    #
    # def rect(self,x,y,w,h,c=[1,1,1,1]):
    #     glBegin(GL_QUADS)
    #     glColor4f(c[0],c[1],c[2],c[3])
    #     glVertex2f(x, y)
    #     glVertex2f(x, y+h)
    #     glVertex2f(x+w, y+h)
    #     glVertex2f(x+w, y)
    #     glEnd()
    #
    # def ellipse(self,x,y,w,h,c=[1,1,1,1],res=16):
    #     glBegin(GL_TRIANGLES)
    #     glColor4f(c[0],c[1],c[2],c[3])
    #     for i in range(res):
    #         glVertex2f(x,y)
    #         glVertex2f(x+math.sin(math.pi*2/res*i)*w,y+math.cos(math.pi*2/res*i)*h)
    #         glVertex2f(x+math.sin(math.pi*2/res*(i+1))*w,y+math.cos(math.pi*2/res*(i+1))*h)
    #     glEnd()
    #
    # def point(self,x,y,c=[1,1,1,1]):
    #     glBegin(GL_POINTS)
    #     glColor4f(c[0],c[1],c[2],c[3])
    #     glVertex2f(x,y)
    #     glEnd()
