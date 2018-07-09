from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
root = tk.Tk()
root.withdraw()

def imload(path):
    img = Image.open(path)
    img = np.float32(img)
    return img
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

src_path = filedialog.askopenfilename(title="Какое фото редактировать?")
color_path = filedialog.askopenfilename(title="На цвета какого фото?")
save_path = filedialog.asksaveasfilename(title="Куда сохранить?",filetypes=[('Джипег','*.jpg')]) + ".jpg"

src = imload(src_path)
color = imload(color_path)

rgb = messagebox.askyesno("Использовать RGB?")
if not rgb:
    luma = messagebox.askyesno("Только цвет?")
    luma = not luma
    src = set_color(src, color, True, luma)
else:
    src = histmatch(src, color)

with open(save_path, 'wb') as file:
    Image.fromarray(np.uint8(np.clip(src,0,255))).save(file, 'jpeg')
