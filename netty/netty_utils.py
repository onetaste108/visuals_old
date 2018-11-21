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
