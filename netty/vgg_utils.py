import numpy as np
vgg_offsets = np.asarray([103.939, 116.779, 123.68])
def preprocess(img):
    x = np.float32(img)
    x = x[...,[2,1,0]]
    x = x - vgg_offsets
    return x
def deprocess(x):
    x = np.uint8(x + vgg_offsets)
    return x[...,[2,1,0]]
def get_bounds(x0):
    bounds = [[- vgg_offsets[0], 255 - vgg_offsets[0]],
          [- vgg_offsets[1], 255 - vgg_offsets[1]],
          [- vgg_offsets[2], 255 - vgg_offsets[2]]] * (x0.size//3)
    return bounds
def get_vgg_shape(input_shape, layer, octave=0, model="vgg16", padding="valid"):
    input_shape = np.array(input_shape)
    shape = input_shape
    for o in range(octave):
        shape = (shape - 5) // 2 + 1
    depth = 3
    if model == "vgg16":
        if layer > 18:
            print("Error num layers")
            return
        for l in range(layer+1):
            if l in [1,2,4,5,7,8,9,11,12,13,15,16,17]:
                if padding == "valid":
                    shape = shape - 2
            elif l in [3,6,10,14,18]:
                shape = shape // 2
        if layer in range(1,4): depth = 64
        elif layer in range(4,7): depth = 128
        elif layer in range(7,11): depth = 256
        elif layer in range(11,19): depth = 512
    elif model == "vgg19":
        if layer > 22:
            print("Error num layers")
            return
        for l in range(layer+1):
            if l in [1,2,4,5,7,8,9,10,12,13,14,15,17,18,19,20]:
                if padding == "valid":
                    shape = shape - 2
            elif l in [3,6,11,16,21]:
                shape = shape // 2
        if layer in range(1,4): depth = 64
        elif layer in range(4,7): depth = 128
        elif layer in range(7,12): depth = 256
        elif layer in range(12,22): depth = 512
    return np.array([shape[0],shape[1],depth])
def get_location(loc,l,model="vgg19"):
    loc = np.int32(loc)
    psize = 1
    if model == "vgg19":
        while l > 0:
            if l in [1,2,4,5,7,8,9,10,12,13,14,15,17,18,19,20]:
                psize += 2
                loc += 0
            elif l in [3,6,11,16,21]:
                psize *= 2
                loc *= 2
            l -= 1
    elif model == "vgg16":
        while l > 0:
            if l in [1,2,4,5,7,8,9,11,12,13,15,16,17]:
                psize += 2
                loc += 0
            elif l in [3,6,10,14,18]:
                psize *= 2
                loc *= 2
            l -= 1
    p_min = 0
    p_max = psize
    location = np.int32([[loc[0]+p_min,loc[0]+p_max],[loc[1]+p_min,loc[1]+p_max]])
    return location
