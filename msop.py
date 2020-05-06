import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

def biliear_interpolation(arr, xs, ys):
    '''
    Input:
        arr: numpy array. image data. shape = (h, w) for grayscale or (h, w, 3) for RGB.
        xs, ys: array-like (list or numpy array). x coordinates and y coordinates.
                                 
                  y= 
                    0   w
                x=0 +---+
                    |   |
                    |   |
                  h +---+   

    Output:
        sampled_image: numpy array. same shape as xs, ys if image is grayscale. for RGB image, the shape would be [xs.shape[0], xs.shape[1], ..., xs.shape[d], 3]
        inside: numpy array. same shape as xs, ys. represents how much the sample is inside the image.
                0: outside the image with 1 pixel margin
                1: inside the image (0 < x < h && 0 < y < w)
                0 < inside < 1: outside but within 1 pixel margin. decays linearly on the edge, bilinearly on the corner.
                    for example: (x,y) = (-0.5, -0.5)   inside = 0.25
                                 (x,y) = (-0.5,  0.5)   inside = 0.5
                                 (x,y) = ( 0.5,  0.5)   inside = 1
                                 (x,y) = (  -1,   -1)   inside = 0
    '''
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    #print(xs < 0)
    #print(type(xs < 0))
    
    #outside = (xs < 0) | (arr.shape[0]-1 < xs) | (ys < 0) | (arr.shape[1]-1 < ys)
    inx = 1 - np.clip(np.maximum(-xs, xs - (arr.shape[0]-1)), 0, 1)
    iny = 1 - np.clip(np.maximum(-ys, ys - (arr.shape[1]-1)), 0, 1)
    inside = inx * iny
    xs = np.clip(xs, 0, arr.shape[0]-1)
    ys = np.clip(ys, 0, arr.shape[1]-1)
    us = np.floor(xs).astype(np.int)
    ls = np.floor(ys).astype(np.int)
    us = np.clip(us, 0, arr.shape[0]-2)
    ls = np.clip(ls, 0, arr.shape[1]-2)
    ds = us + 1
    rs = ls + 1
    
    Iul = arr[us, ls]
    Iur = arr[us, rs]
    Idl = arr[ds, ls]
    Idr = arr[ds, rs]

    wul = (xs-us) * (ys-ls)
    wur = (xs-us) * (rs-ys)
    wdl = (ds-xs) * (ys-ls)
    wdr = (ds-xs) * (rs-ys)
    
    #print(list(xs.shape) + [1] * (len(arr.shape) - 2))
    wul = wul.reshape(list(xs.shape) + [1] * (len(arr.shape) - 2))
    wur = wur.reshape(list(xs.shape) + [1] * (len(arr.shape) - 2))
    wdl = wdl.reshape(list(xs.shape) + [1] * (len(arr.shape) - 2))
    wdr = wdr.reshape(list(xs.shape) + [1] * (len(arr.shape) - 2))
    
    res = Iul * wul + Iur * wur + Idl * wdl + Idr * wdr
    
    return res, inside
    
def msop(img_name, base_lvl = 1):
    '''
    Input:
        img_name: string. image file name
        base_lvl: int. start from which level. 1 means skipping the level with original resolution
    Output:
        feature_xs, feature_ys: list of numpy arrays. feature_xs[0] is the x coodinates of features on level=base_lvl, feature_xs[1] on level=base_lvl+1 ...
        
                  y= 
                    0   w
                x=0 +---+
                    |   |
                    |   |
                  h +---+
                    
        discriptors: list of numpy arrays with shape = (features_count, 8, 8)
    '''
    im = Image.open(img_name)
    im = im.convert('L')
    arrs = [np.asarray(im, dtype=np.float)]
    #print(arrs[0].shape)
    while min(arrs[-1].shape[0], arrs[-1].shape[1]) > 60:
        tmp = arrs[-1][::2, ::2]
        tmp = ndimage.gaussian_filter(tmp, 1.0)
        arrs.append(tmp)
    
    #base_lvl = 1
    lvl = base_lvl
    
    arrs = arrs[base_lvl:]
    
    feature_xs = []
    feature_ys = []
    discriptors = []
    for arr in arrs:
        print('at level {}'.format(lvl))
        print(arr.shape)
        interest_x = []
        interest_y = []
        val = []
        gx, gy = np.gradient(ndimage.gaussian_filter(arr, 1.0))
        gxgx = ndimage.gaussian_filter(gx ** 2, 1.2)
        gxgy = ndimage.gaussian_filter(gx * gy, 1.2)
        gygy = ndimage.gaussian_filter(gy ** 2, 1.2)
        f = (gxgx * gygy - gxgy ** 2) / (gxgx + gygy + 1e-9)
        for i in range(1, f.shape[0]-1):
            for j in range(1, f.shape[1]-1):
                if f[i, j] > 10.0:
                    is_max = True
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            if f[i+di, j+dj] > f[i, j]:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if is_max:
                        grad = np.array([(f[i+1, j] - f[i-1, j]) / 2, (f[i, j+1] - f[i, j-1]) / 2])
                        hessian = np.array([[f[i+1, j] + f[i-1, j] - 2 * f[i, j], (f[i+1, j+1] + f[i-1, j-1] - f[i+1, j-1] - f[i-1, j+1]) / 4],
                                            [(f[i+1, j+1] + f[i-1, j-1] - f[i+1, j-1] - f[i-1, j+1]) / 4, f[i, j+1] + f[i, j-1] - 2 * f[i, j]]])
                        delta = np.linalg.lstsq(hessian, -grad, rcond=None)[0]
                        if np.max(abs(delta)) < 1:
                            v = f[i, j] + np.dot(grad, delta) + 0.5 * delta @ hessian @ delta.T
                            #print(val)
                            interest_x.append(i + delta[0])
                            interest_y.append(j + delta[1])
                            val.append(v)
                        else:
                            #print('big')
                            pass
        
        interest_x = np.asarray(interest_x)
        interest_y = np.asarray(interest_y)
        val = np.asarray(val)
        
        print('feature count: {}'.format(interest_x.shape[0]))
        
        gbx, gby = np.gradient(ndimage.gaussian_filter(arr, 4.5))
        vx = biliear_interpolation(gbx, interest_x, interest_y)[0]
        vy = biliear_interpolation(gby, interest_x, interest_y)[0]
        norm = (vx ** 2 + vy ** 2) ** 0.5
        vx /= norm + 1e-9
        vy /= norm + 1e-9
        sxs = np.linspace(-17.5, 17.5, 8)
        sys = np.linspace(-17.5, 17.5, 8)
        sxs, sys = np.meshgrid(sxs, sys)

        disc_sample = ndimage.gaussian_filter(arr, 2.0)
        disc = np.zeros([interest_x.shape[0], 8, 8])
        stay = np.zeros(interest_x.shape[0], dtype=bool)
        for i in range(interest_x.shape[0]):            
            tmpx = interest_x[i] + vx[i] * sxs - vy[i] * sys
            tmpy = interest_y[i] + vy[i] * sxs + vx[i] * sys
            samples, inside = biliear_interpolation(disc_sample, tmpx, tmpy)
            stay[i] = np.any(inside < 0.8)
            mean, std = np.mean(samples), np.std(samples)
            samples = (samples - mean) / (std + 1e-9)
            disc[i] = samples
            if lvl > 0 and False:
                fig, axes = plt.subplots(ncols=2)
                axes[0].imshow(arr, cmap='gray')
                axes[0].set_xlim(interest_y[i]-40, interest_y[i]+40)
                axes[0].set_ylim(interest_x[i]+40, interest_x[i]-40)
                axes[0].scatter(tmpy, tmpx)
                axes[0].plot([interest_y[i], interest_y[i] + vy[i] * 10], [interest_x[i], interest_x[i] + vx[i] * 10])
                axes[1].imshow(samples, cmap='gray')
                plt.show()
        #print(stay)
        stay = np.logical_not(stay)
        #print(stay)
        interest_x = interest_x[stay]
        interest_y = interest_y[stay]
        disc = disc[stay]
        val = val[stay]
        print('inside feature count: {}'.format(interest_x.shape[0]))

        rs = np.array([float('inf')] * len(interest_x))
        for i in range(len(interest_x)):
            print('getting radius {}/{}'.format(i+1, len(interest_x)), end='\r')    
            for j in range(len(interest_x)):
                if i == j:
                    continue
                if val[i] < 0.9 * val[j]:
                    rs[i] = min(rs[i], ((interest_x[i] - interest_x[j]) ** 2 + (interest_y[i] - interest_y[j]) ** 2) ** 0.5)
        print()
        inds = np.argsort(rs)[-500:]
        #print(inds)
        interest_x = interest_x[inds]
        interest_y = interest_y[inds]
        disc = disc[inds]
        val = val[inds]
        
        '''
        fig, axes = plt.subplots(nrows=2, ncols=3)
        tmpx = interest_x[-1] + vx[-1] * sxs - vy[-1] * sys
        tmpy = interest_y[-1] + vy[-1] * sxs + vx[-1] * sys
        axes[0][0].imshow(gxgx, cmap='gray')
        axes[0][1].imshow(gxgy, cmap='gray')
        axes[0][2].imshow(gygy, cmap='gray')
        axes[1][1].imshow(f, cmap='gray')
        axes[1][1].scatter(interest_y[-1], interest_x[-1])
        axes[1][1].scatter(interest_y[-1] + vy[-1], interest_x[-1] + vx[-1])
        #axes[1][1].scatter(tmpy, tmpx)
        axes[1][2].imshow(arr, cmap='gray')
        axes[1][2].scatter(interest_y[-1], interest_x[-1])
        axes[1][2].scatter(interest_y[-1] + vy[-1], interest_x[-1] + vx[-1])
        plt.show()
        '''
        feature_xs.append(interest_x * 2 ** lvl)
        feature_ys.append(interest_y * 2 ** lvl)
        discriptors.append(disc)
        lvl += 1
        
    for i in range(len(feature_xs)):
        print('{} features at level {}'.format(feature_xs[i].shape[0], i))
        
    return feature_xs, feature_ys, discriptors
if __name__ == '__main__':
    #name = 'parrington/prtn00.jpg'
    name = 'pack2_rot/IMG_0018.JPG'
    feats = msop(name, 1)
    im = Image.open(name)
    arr = np.asarray(im)
    fig, ax = plt.subplots()
    ax.imshow(arr)
    for i in range(len(feats[0])):
        ax.scatter(feats[1][i], feats[0][i])
    plt.show()
    #print(feats[0][0][:30])
    #msop(['download.png'])
    #msop(['pack2_rot/IMG_00{}.JPG'.format(i) for i in range(18, 29)])
