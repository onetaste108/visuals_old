from kivy.app import App
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from time import strftime

class ClockLayout(BoxLayout):
    time_prop = ObjectProperty(None)

class MyApp(App):
    count = 0
    sw_started = False

    def on_start(self):
        Clock.schedule_interval(self.update_time,0)
        Clock.schedule_interval(self.update_clock,0)

    def update_time(self, nap):
        self.root.ids.time.text = strftime("[b]%H[/b]:%M:%S")
        self.root.ids.stopwatch.text = str(self.count)

    def update_clock(self,nap):
        if self.sw_started:
            self.count += 1

    def reset(self):
        if self.sw_started:
            self.root.ids.start_stop.text = "STOP"
        self.sw_started = not self.sw_started

    def start_stop(self):
        self.root.ids.start = ("Start" if self.sw_started else "Stop")

if __name__ == "__main__":

    LabelBase.register(name="Roboto",
    fn_regular="Roboto-Light.ttf",
    fn_bolditalic="Roboto-BoldItalic.ttf",
    fn_italic="Roboto-LightItalic.ttf",
    fn_bold="Roboto-Bold.ttf")

    Window.clearcolor = (.5,.2,.2,1)

    MyApp().run()
