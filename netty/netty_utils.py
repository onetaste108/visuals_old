import numpy as np
def add_tgs(tgs):
    l_tgs = len(tgs)
    l_gram = len(tgs[0])
    out = []
    for i in range(l_gram):
        t = tgs[0][i]
        for j in range(1,l_tgs):
            t += tgs[j][i]
        out.append(t)
    return out

def mix_tgs(tgs,w=None):
    l_tgs = len(tgs)
    l_gram = len(tgs[0])
    if w is not None:
        w = np.float32(w)
        w = w / np.sum(w)
    else:
        w = np.float32([1/l_tgs for i in range(l_tgs)])

    out = []
    for i in range(l_gram):
        t = tgs[0][i] * w[0]
        for j in range(1,l_tgs):
            t += tgs[j][i] * w[j]
        out.append(t)
    return out

def incremental(final=[1024,1024],short=256,step=1.2,maxf=500):
    short = int(short)
    min_f = min(final)
    max_f = max(final)
    long = int(short*(max_f/min_f))
    seq = []
    maxfun = []
    if final[0] > final[1]:
        a = long
        b = short
    else:
        b = long
        a = short
    i = 0
    while True:
        if i == 0:
            maxfun.append(maxf)
        else:
            maxfun.append(int(maxf/pow(step,(i))))
        new = [int(a*pow(step,(i))),int(b*pow(step,(i)))]
        if min(new) >= min_f:
            new = final
            seq.append(new)
            break
        else:
            seq.append(new)
            i += 1
    print(len(seq),"renders ahead.")
    return seq, maxfun
