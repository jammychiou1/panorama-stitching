import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
import math
import msop
import displacement
import projection
#f = 4*837
f = 3.9*837
#f = 705
filenames = []
source = []
for i in range(18,29):
    filenames.append("pack2_rot/IMG_00"+str(i)+".JPG")
#filenames = ['parrington/prtn{}.jpg'.format(str(i).zfill(2)) for i in range(0, 18)]
#print(filenames)
displacements = displacement.displacement(filenames,f)
'''
for i in range(len(filenames)):
    im = Image.open(filenames[i])
    arr = np.array(im)
    source.append(arr)
'''
# for i in range(len(displacements)):
#   displacements[i][0] -= delta*displacements[i][1]
im = Image.open(filenames[0])
arr = np.asarray(im)
img_h = arr.shape[0]
img_w = arr.shape[1]
minx = float('inf')
miny = float('inf')
maxx = -float('inf')
maxy = -float('inf')
delta = (displacements[-1][1] - displacements[0][1])/(displacements[-1][0]-displacements[0][0])
#for i in range(len(displacements)):
#    displacements[i][1] -= displacements[i][0] * delta
for i in range(len(displacements) - 1):
    minx = min(minx, displacements[i][0] - (img_w - 1) / 2)
    maxx = max(maxx, displacements[i][0] + (img_w - 1) / 2)
    miny = min(miny, displacements[i][1] - (img_h - 1) / 2)
    maxy = max(maxy, displacements[i][1] + (img_h - 1) / 2)
print(displacements)
print(minx, maxx, miny, maxy)
'''
for i in range(len(displacements)):
    displacements[i][0] -= minx
    displacements[i][1] -= miny
maxx -= minx
maxy -= miny
'''
center_x = (minx + maxx) / 2
tot_width = np.abs(displacements[-1][0] - displacements[0][1])
print('tot_width', tot_width)
print('expected', f * 2 * np.pi)
#result_x = np.arange(minx, maxx)
result_x = np.linspace(center_x - tot_width / 2, center_x + tot_width / 2, int(tot_width))
result_y = np.arange(maxy, miny, -1)
#result_xs, result_ys = np.meshgrid(result_x, result_y)
result = np.zeros([result_y.shape[0], result_x.shape[0], 3])
weight_sum = np.zeros([result_y.shape[0], result_x.shape[0]])
weight_sum += 1e-9
for i, filename in enumerate(filenames):
    print(filename)
    im = Image.open(filename)
    arr = np.asarray(im) / 255
    xin = (displacements[i][0] - (img_w - 1) / 2 - 3 < result_x) & (result_x < displacements[i][0] + (img_w - 1) / 2 + 3)
    yin = (displacements[i][1] - (img_h - 1) / 2 - 3 < result_y) & (result_y < displacements[i][1] + (img_h - 1) / 2 + 3)
    patch_xs, patch_ys = np.meshgrid(result_x[xin], result_y[yin])
    #print(patch_xs.shape)
    #print(patch_ys.shape)
    patch_xs -= displacements[i][0]
    patch_ys -= displacements[i][1]
    weight = np.maximum(np.minimum(patch_xs + (img_w - 1) / 2, (img_w - 1) / 2 - patch_xs), 0)
    weight **= 3
    #patch_ys -= patch_xs * delta
    patch_xs, patch_ys = projection.planar_projection(patch_xs, patch_ys, f)
    patch_xs, patch_ys = - patch_ys + (img_h - 1) / 2, patch_xs + (img_w - 1) / 2
    patch, inside = msop.bilinear_interpolation(arr, patch_xs, patch_ys)
    weight *= inside
    #print(yin)
    #print(xin)
    for pi, ri in enumerate(np.flatnonzero(yin)):
        for pj, rj in enumerate(np.flatnonzero(xin)):
            #print(ri, rj)
            result[ri, rj] += patch[pi, pj] * weight[pi, pj]
            weight_sum[ri, rj] += weight[pi, pj]
            
    #result[yin, xin] += patch * weight[:, np.newaxis]
    #fig, ax = plt.subplots()
    #ax.imshow(patch * inside[..., np.newaxis])
    #plt.show()
result /= weight_sum[..., np.newaxis]
'''
now = 0
print(result.shape[1])
for j in range(result.shape[1]):
    print(j)
    for i in range(result.shape[0]):
        # for index in range(len(displacements)):
        index = now
        while index < len(displacements):
            i_tmp = (i + delta*j)
            pixel1, inside1 = msop.bilinear_interpolation(source[index%len(source)],[i_tmp-displacements[index][0]],[j-displacements[index][1]])
            if inside1 > 0 and index < len(displacements) - 1:
                pixel2, inside2 = msop.bilinear_interpolation(source[(index+1)%len(source)],[i_tmp-displacements[index+1][0]],[j-displacements[index+1][1]])
                if inside2 > 0:
                    l = displacements[index+1][1]
                    r = displacements[index][1]+(np.tan(source[index%len(source)].shape[1]/2/f)*f-np.tan(-source[index%len(source)].shape[1]/2/f)*f)
                    result[i][j] = pixel1*inside1*(r-j)/(r-l)+pixel2*inside2*(j-l)/(r-l)
                else:
                    result[i][j] = pixel1*inside1
                now = index
                break
            else:
                index = index + 1
'''                
#fig, ax = plt.subplots()
#ax.imshow(result)
#ax.imshow(result, extent=(result_x[0] - 0.5, result_x[-1] + 0.5, result_y[-1] + 0.5, result_y[0] - 0.5))
#ax.plot([minx, maxx], [miny, maxy])
#plt.show()
im = Image.fromarray((result * 255).astype(np.uint8))
im.save('pano.png')

