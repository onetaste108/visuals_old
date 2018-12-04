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

def mix_tgs(tgs):
    l_tgs = len(tgs)
    out = add_tgs(tgs)
    l_gram = len(out)
    for i in range(l_gram):
        out[i] = out[i] / l_tgs
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
            maxfun.append(int(maxf/2/pow(step,(i))))
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
