import numpy as np
from numba import njit
from img_utils import *


class PatchMatch:
    def __init__(self):
        pass

    def setup(self,img1,img2,ksize=1,stride_1=1,stride_2=1):
        self.ksize = ksize
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.iters = 0

        self.img1 = np.int32(img1)
        self.img2 = np.int32(img2)

        self.m1,self.n1 = img1.shape[:2]
        self.m2,self.n2 = img2.shape[:2]
        self.pm1,self.pn1 = (int((self.m1-self.ksize)/stride_1)+1,int((self.n1-self.ksize)/stride_1)+1)
        self.pm2,self.pn2 = (int((self.m2-self.ksize)/stride_2)+1,int((self.n2-self.ksize)/stride_2)+1)

        self.offsets = np.int32(np.random.rand(self.pm1,self.pn1,2)*[self.pm2,self.pn2])
        self.update_best_vals()

    def update_best_vals(self):
        self.best_vals = compute_best_vals(self.img1,self.img2,self.offsets,self.pm1,self.pn1,self.ksize,self.stride_1,self.stride_2)

    def match(self,iters=1):
        for i in range(iters):
            dir = int(self.iters%2==0)*2-1
            iterate(self.img1,self.img2,self.offsets,self.best_vals,dir,self.ksize,self.pm1,self.pn1,self.pm2,self.pn2,self.stride_1,self.stride_2)
            print(np.mean(self.best_vals))
            self.iters += 1

    def brute_match(self):


    def get(self,stride=None):
        if stride is None:
            stride = self.stride_1
        return np.uint8(reconstruct(self.img2,self.offsets,self.ksize,stride,self.stride_2))

    def get_best(self):
        return np.uint8(reconstruct_best(self.img2,self.offsets,self.best_vals,self.ksize,self.stride_1,self.stride_2))

@njit("i4(i4[:,:,:],i4[:,:,:])")
def loss(p1,p2):
    return np.sum(np.square(p1-p2))

# @njit("i4(i4[:,:,:],i4[:,:,:],i8,i8)")
def reconstruct(img,offsets,ksize,stride_1,stride_2):
    pm,pn = offsets.shape[:2]
    m,n = ((pm-1)*stride_1+ksize,(pn-1)*stride_1+ksize)
    out = np.zeros((m,n,3),dtype=np.int32)
    map = np.zeros((m,n,3),dtype=np.int32)
    for i in range(0,pm):
        for j in range(0,pn):
            out[i*stride_1:i*stride_1+ksize,j*stride_1:j*stride_1+ksize] += img[
            offsets[i][j][0]*stride_2:offsets[i][j][0]*stride_2+ksize,
            offsets[i][j][1]*stride_2:offsets[i][j][1]*stride_2+ksize]
            map[i*stride_1:i*stride_1+ksize,j*stride_1:j*stride_1+ksize] += 1
    map[map == 0] = 1
    out = out / map
    return out

def reconstruct_best(img,offsets,best_vals,ksize,stride_1,stride_2):
    pm,pn = offsets.shape[:2]
    m,n = ((pm-1)*stride_1+ksize,(pn-1)*stride_1+ksize)
    out = np.zeros((m,n,3),dtype=np.int32)
    map = np.zeros((m,n,3),dtype=np.int32)
    vals = np.copy(best_vals)
    vals = np.ravel(vals)
    best_ind = np.argsort(vals)
    for i in range(pm*pn):
            ind = best_ind[-i]
            y,x = ind//pn, ind%pn
            out[y*stride_1:y*stride_1+ksize,x*stride_1:x*stride_1+ksize] = img[
            offsets[y][x][0]*stride_2:offsets[y][x][0]*stride_2+ksize,
            offsets[y][x][1]*stride_2:offsets[y][x][1]*stride_2+ksize]
    return out

@njit("void(i4[:,:,:],i4[:,:,:],i4[:,:,:],i4[:,:],i8,i8,i8,i8,i8,i8,i8,i8,i8,i8)")
def propagate(img1,img2,offsets,best_vals,y,x,dir,ksize,pm1,pn1,pm2,pn2,stride_1,stride_2):
    ca = np.empty((3,2),dtype=np.int32)
    count = 1
    dir_2 = max(1,int(dir/stride_2*stride_1+0.5))
    # dir_2 = dir

    if y-dir >= 0 and y-dir < pm1:
        yo = offsets[y-dir][x]+np.int32([dir_2,0])
        if yo[0] >= 0 and yo[0] < pm2:
            ca[count] = yo
            count += 1

    if x-dir >= 0 and x-dir < pn1:
        xo = offsets[y][x-dir]+np.int32([0,dir_2])
        if xo[1] >=0 and xo[1] < pn2:
            ca[count] = xo
            count += 1

    if count > 1:
        ca[0] = offsets[y][x]
        cav = np.empty((count,),dtype=np.int32)
        cav[0] = best_vals[y,x]

        for i in range(1,count):
            cav[i] = loss(
                img1[y*stride_1:y*stride_1+ksize,x*stride_1:x*stride_1+ksize],
                img2[ca[i][0]*stride_2:ca[i][0]*stride_2+ksize,ca[i][1]*stride_2:ca[i][1]*stride_2+ksize])

        best = np.argmin(cav)
        if best > 0:
            offsets[y][x] = ca[best]
            best_vals[y][x] = cav[best]


@njit("void(i4[:,:,:],i4[:,:,:],i4[:,:,:],i4[:,:],i8,i8,i8,i8,i8,i8,i8,i8,i8)")
def randomize(img1,img2,offsets,best_vals,y,x,ksize,pm1,pn1,pm2,pn2,stride_1,stride_2):
    w = max(pm2,pn2)
    a = .5
    i = 0
    frame_size = w*pow(a,i)
    best_val = best_vals[y][x]
    best_offset = offsets[y][x]
    orig = best_offset.copy()
    while(frame_size >= 1):
        frame_pos = np.array([max(0,orig[0]-frame_size),max(0,orig[1]-frame_size)])
        frame_max = np.array([min(orig[0]+frame_size,pm2),min(orig[1]+frame_size,pn2)])
        frame = frame_max-frame_pos
        no = ((np.random.rand((1)) * frame) + frame_pos).astype(np.int32)
        new_val = loss(
            img1[y*stride_1:y*stride_1+ksize,x*stride_1:x*stride_1+ksize],
            img2[no[0]*stride_2:no[0]*stride_2+ksize,no[1]*stride_2:no[1]*stride_2+ksize])

        if new_val < best_val:
            best_val = new_val
            best_offset = no
        i+=1
        frame_size = w*pow(a,i)

    best_vals[y][x] = best_val
    offsets[y][x] = best_offset

@njit("void(i4[:,:,:],i4[:,:,:],i4[:,:,:],i4[:,:],i8,i8,i8,i8,i8,i8,i8,i8)")
def iterate(img1,img2,offsets,best_vals,dir,ksize,pm1,pn1,pm2,pn2,stride_1,stride_2):
    if dir > 0:
        r = [0,pm1,0,pn1]
    else:
        r = [pm1-1,1,pn1-1,1]
    for i in range(r[0],r[1],dir):
        for j in range(r[2],r[3],dir):
            propagate(img1,img2,offsets,best_vals,i,j,dir,ksize,pm1,pn1,pm2,pn2,stride_1,stride_2)
            randomize(img1,img2,offsets,best_vals,i,j,ksize,pm1,pn1,pm2,pn2,stride_1,stride_2)

# @njit("i4[:,:](i4[:,:,:],i4[:,:,:],i4[:,:,:],i8,i8,i8)")
def compute_best_vals(img1,img2,offsets,pm1,pn1,ksize,stride_1,stride_2):
        best_vals = np.empty((pm1,pn1),dtype=np.int32)
        for y in range(pm1):
            for x in range(pn1):
                best_vals[y][x] = loss(
                    img1[y*stride_1:y*stride_1+ksize,x*stride_1:x*stride_1+ksize],
                    img2[offsets[y][x][0]*stride_2:offsets[y][x][0]*stride_2+ksize,
                        offsets[y][x][1]*stride_2:offsets[y][x][1]*stride_2+ksize])
        return best_vals
