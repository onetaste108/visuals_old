from PIL import Image
import numpy as np
def imload(path):
    img = Image.open(path)
    img = np.uint8(img)
    return img
def imsize(img,size=None,max_size=None,factor=None,mode=Image.BILINEAR):
    img = Image.fromarray(np.clip(img,0,255).astype("uint8"))
    if size is not None:
        im_ratio = img.size[0] / img.size[1]
        crop_ratio = size[0] / size[1]
        if im_ratio < crop_ratio:
            crop_factor = size[0] / img.size[0]
        else:
            crop_factor = size[1] / img.size[1]
        new_size = np.array(img.size) * crop_factor
        new_middle = new_size / 2
        new_padding = (new_middle[0] - size[0] / 2,
                       new_middle[1] - size[1] / 2,
                       new_middle[0] + size[0] / 2,
                       new_middle[1] + size[1] / 2)
        img = img.resize(tuple(np.ceil(new_size).astype("int32")), mode)
        img = img.crop(new_padding)
    elif factor is not None:
        new_size = np.array(img.size[:2]) * factor
        img = img.resize(tuple(np.floor(new_size).astype("int32")), mode)
    return np.uint8(img)
def impropscale(src,tar):
    return np.sqrt((tar[0]*tar[1]) / (src[0]*src[1]))
def imshow(img):
    if ifip(): display(Image.fromarray(np.uint8(img)))
def imsave(img,path):
    Image.fromarray(np.uint8(img)).save(path)

def histmatch(src, color):
    new = np.zeros(src.shape)
    for ch in range(src.shape[-1]):
        src_ch = src[:,:,ch]
        color_ch = color[:,:,ch]
        oldshape = src_ch.shape
        source = src_ch.ravel()
        template = color_ch.ravel()
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        new[:,:,ch] = interp_t_values[bin_idx].reshape(oldshape)
    return new
def set_color(src, color, hist=True, luma=True):
    src = np.array(Image.fromarray(np.uint8(np.clip(src,0,255))).convert('YCbCr'))
    color = np.array(Image.fromarray(np.uint8(np.clip(color,0,255))).convert('YCbCr'))
    if hist:
        if luma:
            src[:,:,:] = histmatch(src[:,:,:],color[:,:,:])
        else:
            src[:,:,1:] = histmatch(src[:,:,1:],color[:,:,1:])
    else:
        src[:, :, 1:] = color[:, :, 1:]
    src = np.float32(Image.fromarray(src, mode='YCbCr').convert('RGB'))
    return src





def ifip():
    try:
        cfg = get_ipython().config
        return True
    except NameError: return False
