import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
import math
import msop
import displacement
import projection
f = 4*837
filenames = []
source = []
for i in range(18,29):
    filenames.append("pack2_rot/IMG_00"+str(i)+".JPG")
displacements = displacement.displacement(filenames,f)
delta = (displacements[-1][1] - displacements[0][1])/(displacements[-1][0]-displacements[0][0])
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
minx = float(inf)
miny = float(inf)
maxx = -float(inf)
maxy = -float(inf)
for i in range(len(displacements)):
    minx = min(minx, displacements[i][0] - (img_w - 1) / 2)
    maxx = max(maxx, displacements[i][0] + (img_w - 1) / 2)
    miny = min(miny, displacements[i][1] - (img_h - 1) / 2)
    maxy = max(maxy, displacements[i][1] + (img_h - 1) / 2)
'''
for i in range(len(displacements)):
    displacements[i][0] -= minx
    displacements[i][1] -= miny
maxx -= minx
maxy -= miny
'''
center_x = (minx + maxx) / 2
result_x = np.linspace(center_x - f * np.pi, center_x + f * np.pi, f * 2 * np.pi)
result_y = np.arange(maxy, miny, -1)
#result_xs, result_ys = np.meshgrid(result_x, result_y)
result = np.zeros([result_y.shape[0], result_x.shape[0]])
weight_sum = np.zeros([result_y.shape[0], result_x.shape[0]])
weight_sum += 1e-9
for i, filename in enumerate(filenames):
    print(filename)
    im = Image.open(filename)
    arr = np.asarray(im) / 255
    xin = (displacements[i][0] - (img_w - 1) / 2 - 3 < result_x) & (result_x < displacements[i][0] + (img_w - 1) / 2 + 3)
    yin = (displacements[i][1] - (img_h - 1) / 2 - 3 < result_y) & (result_y < displacements[i][1] + (img_h - 1) / 2 + 3)
    patch_xs, patch_ys = np.meshgrid(result_x[xin], result_y[yin])
    weight = np.minimum(patch_xs + (img_w - 1) / 2, (img_w - 1) / 2 - patch_xs)
    patch_xs -= displacements[i][0]
    patch_ys -= displacements[i][1]
    patch_ys -= patch_xs * delta
    patch_xs, patch_ys = projection.planar_projection(patch_xs, patch_ys, focal_length)
    patch_xs, patch_ys = - patch_ys + (img_h - 1) / 2, patch_xs + (img_w - 1) / 2
    patch, inside = msop.bilinear_interpolation(arr, patch_xs, patch_ys)
    weight *= inside
    result[yin, xin] += patch * weight
    weight_sum += weight
    plt.imshow(patch)
    plt.show()
result /= weight_sum
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
plt.imshow(result)
plt.show()


