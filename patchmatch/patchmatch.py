import numpy as np
from img_utils import *

class PatchMatch:
    def __init__(self):
        pass

    def match(self,img1,img2,ksize,iters=1):
        img1 = np.int32(img1)
        img2 = np.int32(img2)

        m1,n1 = img1.shape[:2]
        m2,n2 = img2.shape[:2]
        pm1,pn1 = (m1-ksize+1,n1-ksize+1)
        pm2,pn2 = (m2-ksize+1,n2-ksize+1)

        offsets = np.int32(np.random.rand(pm1,pn1,2)*[pm2,pn2])

        for i in range(iters):
            print("iter",i)
            self.iterate(img1,img2,offsets,ksize,pm1,pn1,pm2,pn2,i%2==0)
            imshow(self.reconstruct(img2,offsets,ksize))

    def iterate(self,img1,img2,offsets,ksize,pm1,pn1,pm2,pn2,even=False):
        if even:
            i1,i2,j1,j2,of = (pm1-2,0,pn1-2,0,-1)
        else:
            i1,i2,j1,j2,of = (1,pm1,1,pn1,1)

        for i in range(i1,i2,of):
            for j in range(i1,i2,of):
                # self.prop(img1,img2,offsets,i,j,of,ksize,pm1,pn1,pm2,pn2)
                self.rand(img1,img2,offsets,i,j,of,ksize,pm1,pn1,pm2,pn2)

    def prop(self,img1,img2,offsets,y,x,of,ksize,pm1,pn1,pm2,pn2):
        lo = np.minimum([pm2-1,pn2-1],np.maximum([0,0],offsets[y-of][x]+[of,0]))
        to = np.minimum([pm2-1,pn2-1],np.maximum([0,0],offsets[y][x-of]+[0,of]))
        co = offsets[y][x]

        lv = self.loss(
            img1[y:y+ksize,x:x+ksize],
            img2[lo[0]:lo[0]+ksize,lo[1]:lo[1]+ksize])
        tv = self.loss(
            img1[y:y+ksize,x:x+ksize],
            img2[to[0]:to[0]+ksize,to[1]:to[1]+ksize])
        cv = self.loss(
            img1[y:y+ksize,x:x+ksize],
            img2[co[0]:co[0]+ksize,co[1]:co[1]+ksize])
        offsets[y][x] = [lo,to,co][np.argmin([lv,tv,cv])]

    def rand(self,img1,img2,offsets,y,x,of,ksize,pm1,pn1,pm2,pn2):
        i = 0
        w = max(pm2,pn2)
        a = 1/2

        best_offset = offsets[y][x]
        co = offsets[y][x]
        best_val = self.loss(
            img1[y:y+ksize,x:x+ksize],
            img2[co[0]:co[0]+ksize,co[1]:co[1]+ksize])

        while(w*pow(a,i) >= 1):
            r = np.random.rand((2))*2-1
            no = np.int32(w*pow(a,i)*r)
            no = np.minimum([pm2-1,pn2-1],np.maximum([0,0],no+[y,x]))
            new_val = self.loss(
                img1[y:y+ksize,x:x+ksize],
                img2[no[0]:no[0]+ksize,no[1]:no[1]+ksize])
            if new_val < best_val:
                best_val = new_val
                best_offset = no

            i+=1

        offsets[y][x] = best_offset

    def loss(self,p1,p2):
        return(np.sum(np.square(p1-p2)))

    def reconstruct(self,img,offsets,ksize):
        pm,pn = offsets.shape[:2]
        m,n = (pm+ksize-1,pn+ksize-1)
        out = np.zeros((m,n,3),dtype=np.int32)

        for i in range(pm):
            for j in range(pn):
                out[i:i+ksize,j:j+ksize] += img[
                offsets[i][j][0]:offsets[i][j][0]+ksize,
                offsets[i][j][1]:offsets[i][j][1]+ksize]

        for i in range(m):
            for j in range(n):
                val = min(i+1,m-i,ksize)*min(j+1,n-j,ksize)
                out[i][j] = out[i][j] / val

        return np.uint8(out)
