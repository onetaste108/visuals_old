from kivy.app import App
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex
from kivy.config import Config
from kivy.graphics import Color, Line, Callback

Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "720")
Config.set("graphics", "resizable", "0")

from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.togglebutton import ToggleButton

class RadioButton(ToggleButton):
    def _do_press(self):
        if self.state == "normal":
            ToggleButtonBehavior._do_press(self)

class CanvasWidget(Widget):
    def on_touch_down(self,touch):
        if Widget.on_touch_down(self,touch): return
        with self.canvas:
            pass
            Color(*get_color_from_hex("#23847280"))
            touch.ud["line"] = Line(points=(touch.x,touch.y), width=10)

    def on_touch_move(self, touch):
        if "line" in touch.ud:
            touch.ud["line"].points += (touch.x,touch.y)

    def clear_canvas(self):
        saved = self.children[:]
        self.clear_widgets()
        self.canvas.clear()
        for w in saved: self.add_widget(w)

class PaintApp(App):
    def build(self):
        return CanvasWidget()



if __name__ == "__main__":
    from kivy.core.window import Window

    Window.clearcolor = get_color_from_hex("#010101")

    PaintApp().run()
