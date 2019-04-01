from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.instructions import RenderContext
from kivy.graphics import Rectangle, Color, Mesh, ClearBuffers
from kivy.clock import Clock
from numpy.random import rand

# Mesh(vertices=(0,0,0,.5,.5,0,.5,.5),indices=(0,1,2,3),
#      mode="triangle_strip",
#      fmt=[(b'pos', 2, "float")])

vs = """
attribute vec2 vPosition;
void main()
{
    gl_Position = vec4(vPosition, 0.0, 1.0);
}
"""
fs = """
void main()
{
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""


class GLWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = RenderContext()
        self.prog.shader.vs = vs
        self.prog.shader.fs = fs
        self.canvas = self.prog

    def draw(self):
        self.canvas.draw()

class MainApp(App):
    def build(self):
        self.glw = GLWidget()
        return self.glw

    def draw(self,dt):
        self.glw.canvas.clear()
        with self.glw.canvas:
            ClearBuffers(clear_color=False)
            Rectangle(pos=(rand()*2-1,rand()*2-1),size=(0.01,0.01))

    def on_start(self):
        Clock.schedule_interval(self.draw,0)

if __name__ == "__main__":
    MainApp().run()
