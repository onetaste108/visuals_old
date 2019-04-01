from kivy.app import App
from kivy.base import EventLoop
from kivy.graphics.instructions import RenderContext
from kivy.graphics import Rectangle, BindTexture, Fbo, Callback
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.clock import Clock

import numpy as np
from PIL import Image

T_SIZE = (1024,1024)
W_SIZE = (720,1280)

IMG = np.uint8(Image.open("data/test.jpg").resize(T_SIZE)).tobytes()

from kivy.config import Config
Config.set("graphics", "width", str(W_SIZE[0]))
Config.set("graphics", "height", str(W_SIZE[1]))

class ShaderWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Tex1 = Texture.create(size=T_SIZE, colorfmt="rgba", bufferfmt="float")
        Tex2 = Texture.create(size=T_SIZE, colorfmt="rgba", bufferfmt="float")
        Tex3 = Texture.create(size=T_SIZE, colorfmt="rgba", bufferfmt="float")
        Tex4 = Texture.create(size=T_SIZE, colorfmt="rgba", bufferfmt="float")
        Tex4.mag_filter = "linear"

        self.fbo_edit = Fbo(clear_color=(0.5,0.5,0,1),size=(T_SIZE),texture=Tex1)
        self.fbo_edit.shader.vs = open("data/shaders/vs.glsl").read()
        self.fbo_edit.shader.fs = open("data/shaders/fs_edit.glsl").read()
        self.fbo_edit["viewport"] = [float(s) for s in W_SIZE]
        self.fbo_edit["mode"] = 0

        self.fbo_mix = Fbo(clear_color=(0.5,0.5,0,1),size=(T_SIZE),texture=Tex2)
        self.fbo_mix.shader.vs = open("data/shaders/vs.glsl").read()
        self.fbo_mix.shader.fs = open("data/shaders/fs_mix.glsl").read()
        self.fbo_mix["tex1"] = 1
        self.fbo_mix["tex2"] = 2
        self.fbo_mix["viewport"] = [float(s) for s in W_SIZE]

        self.fbo_save = Fbo(clear_color=(0.5,0.5,0,1),size=(T_SIZE),texture=Tex3)
        self.fbo_save.shader.vs = open("data/shaders/vs.glsl").read()
        self.fbo_save.shader.fs = open("data/shaders/fs_save.glsl").read()
        self.fbo_save["tex"] = 3
        self.fbo_save["viewport"] = [float(s) for s in W_SIZE]

        self.fbo_warp = Fbo(size=(T_SIZE),texture=Tex4)
        self.fbo_warp.shader.vs = open("data/shaders/vs.glsl").read()
        self.fbo_warp.shader.fs = open("data/shaders/fs_warp.glsl").read()
        self.fbo_warp["tex"] = 4
        self.fbo_warp["warp"] = 5
        self.fbo_warp["viewport"] = [float(s) for s in W_SIZE]

        self.tex = Texture.create(size=T_SIZE,colorfmt="rgb",bufferfmt="ubyte")
        self.tex.blit_buffer(IMG)



    def draw(self,nap):
        self.canvas.clear()
        self.canvas.add(self.fbo_edit)
        self.canvas.add(self.fbo_mix)
        self.canvas.add(self.fbo_save)
        self.canvas.add(self.fbo_warp)
        self.fbo_mix.add(self.fbo_edit)
        self.fbo_warp.add(self.fbo_mix)
        with self.canvas:

            with self.fbo_edit:
                Rectangle(pos=(-1,-1),size=(2,2))

            with self.fbo_save:
                BindTexture(texture=self.fbo_mix.texture, index=3)

            with self.fbo_mix:
                BindTexture(texture=self.fbo_save.texture, index=1)
                BindTexture(texture=self.fbo_edit.texture, index=2)
                Rectangle(pos=(-1,-1),size=(2,2))

            with self.fbo_warp:
                BindTexture(texture=self.tex,index=4)
                BindTexture(texture=self.fbo_mix.texture,index=5)
                Rectangle(pos=(-1,-1),size=(2,2))

            Rectangle(pos=(0,0),size=W_SIZE,texture=self.fbo_warp.texture)

    def on_touch_down(self,touch):
        self.fbo_edit["mode"] = 1
        self.fbo_edit["touch_begin"] = (touch.x,touch.y)
        self.fbo_edit["touch_end"] = (touch.x,touch.y)
        self.draw(0)

    def on_touch_move(self, touch):
        self.fbo_edit["touch_end"] = (touch.x,touch.y)

    def on_touch_up(self, touch):
        with self.fbo_save:
            Rectangle(pos=(-1,-1),size=(2,2))
    # self.fbo_edit["mode"] = 0

class ShaderApp(App):
    def on_start(self):
        glwidget = self.root
        Clock.schedule_interval(glwidget.draw,0)



if __name__ == "__main__":
    EventLoop.ensure_window()
    ShaderApp().run()
