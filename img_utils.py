from PIL import Image
import numpy as np
def load(path):
    img = Image.open(path)
    img = np.uint8(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img,-1)
    if img.shape[-1] == 1:
        img = np.repeat(img,3,-1)
    if img.shape[-1] > 3:
        img = img[...,:3]
    return img
def size(img,size=None,max_size=None,factor=None,mode=Image.BILINEAR):
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
def propscale(src,tar):
    return np.sqrt((tar[0]*tar[1]) / (src[0]*src[1]))
def show(img):
    if ifip(): display(Image.fromarray(np.uint8(img)))
def prev(img):
    show(size(img,factor=propscale(img.shape[:2],[256,256])))
def save(img,path):
    Image.fromarray(np.uint8(img)).save(path)
def save_frame(image, path):
    import os
    save = np.copy(image)
    save_path = path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num = len(os.listdir(save_path))
    save_path = os.path.join(save_path, "frame"+"00000"[len(str(num)):]+str(num)+".png")
    with open(save_path, 'wb') as file:
        Image.fromarray(save).save(file)

def histmatch(src, color, m1=None, m2=None):
    new = np.zeros(src.shape)
    for ch in range(src.shape[-1]):
        src_ch = src[:,:,ch]
        color_ch = color[:,:,ch]
        oldshape = src_ch.shape
        source = src_ch.ravel()
        source_backup=source
        if m1 is not None:
            m_1 = size(m1,[src.shape[1],src.shape[0]])
            m_1 = np.int32(np.float32(m_1[:,:,0]).ravel()/255+0.5).astype(np.bool)
            source = source[m_1]
        template = color_ch.ravel()
        if m2 is not None:
            m_2 = size(m2,[color.shape[1],color.shape[0]])
            m_2 = np.int32(np.float32(m_2[:,:,0]).ravel()/255+0.5).astype(np.bool)
            template = template[m_2]
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        if m1 is not None:
            source_backup[m_1] = interp_t_values[bin_idx]
        else:
            source_backup = interp_t_values[bin_idx]
        new[:,:,ch] = source_backup.reshape(oldshape)
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


def rot(img):
    out = [img]
    for i in range(1,4):
        out.append(np.rot90(img,i))
    return out

def bw(img):
    img = np.copy(img)
    bw=img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3
    img[:,:,0],img[:,:,1],img[:,:,2] = bw,bw,bw
    return img


def ifip():
    try:
        cfg = get_ipython().config
        return True
    except NameError: return False
