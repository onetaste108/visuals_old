import numpy as np
from netty import module_style
from netty import netty_utils as nutil

def split(img,ksize,overlay):
    m,n = img.shape[:2]
    pm = (m-ksize)//(ksize-2-overlay)+2
    pn = (n-ksize)//(ksize-2-overlay)+2
    overm = (m-ksize)/(pm-1)
    overn = (n-ksize)/(pm-1)
    laym = (ksize-overm)-overlay
    layn = (ksize-overn)-overlay
    patches = np.empty((pm,pn,2,2),dtype=np.int32)
    for i in range(pm):
        for j in range(pn):
            if i == 0 or i == pm-1:
                moff = 0
            else:
                moff = np.random.rand()*laym - laym/2
            if j == 0 or j == pn-1:
                noff = 0
            else:
                noff = np.random.rand()*layn - layn/2
            mloc = i*overm+moff
            nloc = j*overn+noff
            patches[i][j] = [[mloc,mloc+ksize],[nloc,nloc+ksize]]
    return patches

def split_ref(img,ksize,stride):
    m,n = img.shape[:2]
    pm,pn = (int((m-ksize)/stride)+1,int((n-ksize)/stride)+1)
    patches = np.empty((pm,pn,2,2),dtype=np.int32)
    for i in range(pm):
        for j in range(pn):
            patches[i][j] = [[i*stride,i*stride+ksize],[j*stride,j*stride+ksize]]
    return patches

def compute_grams(img,mask,patches,model):
    m,n = patches.shape[:2]
    grams = []
    for i in range(m):
        grams.append([])
        for j in range(n):
            patch = img[patches[i][j][0][0]:patches[i][j][0][1],patches[i][j][1][0]:patches[i][j][1][1]]
            gram = model.predict([np.float32([patch])]+mask)
            grams[-1].append(gram)
    return grams

def gram_loss(gram1,gram2):
    loss = 0
    for g1, g2 in zip(gram1, gram2):
        l = np.sum(np.square(g1-g2))
        loss += l
    return loss

def find_best(g,grams):
    m,n = len(grams),len(grams[0])
    best = np.inf
    best_id = [0,0]
    best_gram = []
    for i in range(m):
        for j in range(n):
            l = gram_loss(g,grams[i][j])
            if l < best:
                best = l
                best_id = [i,j]
                best_gram = grams[i][j]
    return best_id, best_gram

def match_grams(grams1,grams2):
    m,n = len(grams1),len(grams1[0])
    best_ids = np.zeros((m,n,2),dtype=np.int32)
    best_grams = []
    for i in range(m):
        best_grams.append([])
        for j in range(n):
            best_ids[i][j], bg = find_best(grams1[i][j],grams2)
            best_grams[-1].append(bg)
    return best_ids, best_grams

def match(src,src_mask,imgs,img_masks,model,ksize=512,overlay=0,ref_stride=256):
    patches_src = split(src,ksize,overlay)
    patches_imgs = []
    for img in imgs:
        patches_imgs.append(split_ref(img,ksize,ref_stride))
    grams_src = compute_grams(src,src_mask,patches_src,model)
    grams_imgs = []
    for img, mask, patch in zip(imgs,img_masks,patches_imgs):
        grams_src.append(compute_grams(img,mask,patch,model))
    best_grams = []
    for gram in grams_imgs:
        best_ids, best_gram = match_grams(grams_src,gram)
        best_grams.append(best_gram)
    return patches_1, best_grams
