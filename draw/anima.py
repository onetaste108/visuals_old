import numpy as np

class Anima:
    def __init__(self):
        self.tracks = []
    def track(self, val=None, loop=0):
        t = Track(val,loop)
        self.tracks.append(t)
        return t
    def update(self, time):
        for t in self.tracks:
            t.update(time)

class Track:
    def __init__(self, val = None, loop = 0):
        self.val = val
        self.kfs = []
        self.loop = loop

    def set_kf(self, time, val):
        kf = Keyframe(time, val)
        if len(self.kfs) == 0:
            self.kfs.append(kf)
            return
        id = 0
        while True:
            if id < len(self.kfs):
                if self.kfs[id].time < time: id += 1
                else:
                    self.kfs.insert(id,kf)
                    break
            else:
                self.kfs.append(kf)
                break

    def set(self, val):
        self.val = val

    def get(self):
        return self.val

    def update(self, time):
        if len(self.kfs) == 0:
            pass
        elif len(self.kfs) == 1:
            self.val = self.kfs[0].val
        else:
            if self.loop == 1:
                t0 = self.kfs[0].time
                t1 = self.kfs[-1].time
                time = t0 + ((time-t0) % (t1-t0))

            if time <= self.kfs[0].time:
                self.val = self.kfs[0].val
            elif time >= self.kfs[-1].time:
                self.val = self.kfs[-1].val
            else:
                id = 0
                while (self.kfs[id].time <= time): id += 1
                self.val = kf_linear(self.kfs[id-1], self.kfs[id], time)


class Keyframe:
    def __init__(self,time,val):
        self.time = time
        self.val = val

def kf_linear(k1,k2,time):
    return k1.val + (k2.val - k1.val) * ((time - k1.time) / (k2.time - k1.time))
