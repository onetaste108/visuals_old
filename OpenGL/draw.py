import moderngl
from pyglet.gl import *
import numpy as np

class Draw:
    def __init__(self):
        self.ctx = moderngl.create_context();
        glEnable(GL_BLEND)
        self.prog = self.ctx.program(
            vertex_shader=open("sd2d_v.glsl").read(),
            fragment_shader=open("sd2d_circle.glsl").read())
        self.vbo = self.ctx.buffer(np.float32([-1.0, -1.0,
                                               -1.0,  1.0,
                                                1.0, -1.0,
                                                1.0,  1.0]).tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog,self.vbo,"position")

        self._color = np.float32([1.0, 1.0, 1.0, 1.0])
        self.is_stroke = False
        self.stroke_weight = 0.02

    def color(self, color):
        self._color = np.float32(color)

    def stroke(self):
        self.is_stroke = True

    def no_stroke(self):
        self.is_stroke = False

    def stroke_weight(self, w):
        self.stroke_weight = w

    def sd2d_circle(self,x=0,y=0,r=1):
        self.prog["color"].value = tuple(self._color)
        self.prog["center"].value = (x,y)
        self.prog["radius"].value = r
        self.prog["is_stroke"].value = self.is_stroke
        self.prog["stroke_weight"].value = 0.01
        self.vao.render(GL_TRIANGLE_STRIP)

    # def sd2d_rect(self,x=0,y=0,w=1,h=1):
    #     self.prog["color"].value = tuple(self._color)
    #     self.prog["center"].value = (x,y)
    #     self.prog["radius"].value = r
    #     self.prog["is_stroke"].value = self.is_stroke
    #     self.prog["stroke_weight"].value = 0.01
    #     self.vao.render(GL_TRIANGLE_STRIP)
