from img_utils import *
from scipy import ndimage

def get_energy(img):
#     g = np.gradient(img)
#     return np.sqrt(np.abs(np.sum(g[1],axis=-1))**2+np.abs(np.sum(g[1],axis=-1))**2)

    # Get x-gradient in "sx"
    sx = ndimage.sobel(img,axis=0,mode='constant')
    # Get y-gradient in "sy"
    sy = ndimage.sobel(img,axis=1,mode='constant')
    # Get square root of sum of squares
    sobel=np.hypot(sx,sy)

    return np.sum(sobel,axis=2)

def calc_cost(en):
    m, n = en.shape
    tab = en.copy()
    for y in range(1,m):
        for x in range(n):
            tab[y,x] = en[y,x] + np.amin(tab[y-1,max(0,x-1):min(n,x+2)])
    return tab

def find_path(tab,id=None):
    m,n = tab.shape
    path = np.zeros(m,dtype=np.int32)
    if not id:
        path[-1] = np.argmin(tab[-1])
    else:
        path[-1] = id
    for y in range(m-2,-1,-1):
        lb = max(0,path[y+1]-1)
        rb = min(path[y+1]+2,n)
        sl = tab[y,lb:rb]
        off = np.argmin(sl)
        if path[y+1] == 0:
            path[y] = off
        else:
            path[y] = path[y+1]-1+off
    return path

def draw_path(img,path,color=[255,0,0]):
    draw = img.copy()
    h,w,c = draw.shape
    for y in range(h):
        draw[y,path[y]] = color
    return draw

def carve_path(img,path):
    m, n = img.shape[: 2]
    out = np.zeros((m, n - 1, 3))
    for y in range(m):
        out[y, :, 0] = np.delete(img[y, :, 0], path[y])
        out[y, :, 1] = np.delete(img[y, :, 1], path[y])
        out[y, :, 2] = np.delete(img[y, :, 2], path[y])
    return out

def add_path(img,path):
    m,n = img.shape[:2]
    out = np.zeros((m, n + 1, 3))
    for y in range(m):
        out[y, :path[y]+1] = img[y,:path[y]+1]
        out[y,path[y]+1] = img[y,path[y]]
        if path[y]+1 <= n+1:
            out[y,path[y]+2:] = img[y,path[y]+1:]
    return out

def sort_paths(tab):
    m,n = tab.shape
    values = tab[-1].copy()
    sorted = np.arange(n)
    for i in range(n):
        best_n = values[i]
        best_id = i
        for j in range(i,n,1):
            if best_n > values[j]:
                best_n = values[j]
                best_id = j
        temp_id = sorted[best_id]
        temp_val = values[best_id]
        sorted[best_id] = sorted[i]
        sorted[i] = temp_id
        values[best_id] = values[i]
        values[i] = temp_val
    return sorted
