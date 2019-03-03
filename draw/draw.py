import moderngl
from pyglet.gl import *
import numpy as np

ctx = None

class Draw:
    def __init__(self):
        global ctx
        ctx = moderngl.create_context()
        glEnable(GL_BLEND)

        self.sd2d = SD2D()

    def clear(self):
        ctx.clear()

    def viewport(self,w,h):
        self.sd2d.viewport(w,h)


class SD2D:
    def __init__(self):
        self.prog = load_shaders("sd2d_v","sd2d_f")
        self.vbo = quad_vbo()
        self.vao = ctx.simple_vertex_array(self.prog, self.vbo, "pos")

        self.COLOR = None
        self.STROKE_WEIGHT = None
        self.COL_MODE = None
        self.SMOOTH = None

        self.FILL = 0
        self.STROKE = 1

        self.CIRCLE = 0
        self.LINE = 1


        self.viewport(1,1)
        self.color([1.0,1.0,1.0,1.0])
        self.stroke_weight(0.01)
        self.smooth(0)
        self.fill()


    def circle(self,x,y,r):
        self.prog["SHAPE"].value = self.CIRCLE
        self.prog["center"].value = (x,y)
        self.prog["radius"].value = r

        self.vao.render(GL_TRIANGLE_STRIP)

    def line(self,x1,y1,x2,y2):
        self.prog["SHAPE"].value = self.LINE
        self.prog["p1"].value = (x1,y1)
        self.prog["p2"].value = (x2,y2)

        self.vao.render(GL_TRIANGLE_STRIP)

    def viewport(self,w,h):
        self.prog["screen_size"].value = (w,h)

    def color(self,c):
        self.COLOR = np.float32(c)
        self.prog["color"].value = tuple(self.COLOR)

    def stroke_weight(self,w):
        self.STROKE_WEIGHT = w
        self.prog["stroke_weight"].value = self.STROKE_WEIGHT

    def fill(self):
        self.COL_MODE = self.FILL
        self.prog["COL_MODE"].value = self.COL_MODE

    def stroke(self):
        self.COL_MODE = self.STROKE
        self.prog["COL_MODE"].value = self.COL_MODE

    def smooth(self,w):
        self.SMOOTH = w
        self.prog["aa"].value = self.SMOOTH



def load_shaders(v,f):
    return ctx.program(vertex_shader=open("shaders/"+v+".glsl").read(),
                       fragment_shader=open("shaders/"+f+".glsl").read())

def quad_vbo():
    return ctx.buffer(np.float32([-1.0, -1.0,
                                  -1.0,  1.0,
                                   1.0, -1.0,
                                   1.0,  1.0]).tobytes())
