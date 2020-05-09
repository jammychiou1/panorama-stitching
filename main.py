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
delta = (displacements[-1][0] - displacements[0][0])/(displacements[-1][1]-displacements[0][1])
for i in range(len(filenames)):
	im = Image.open(filenames[i])
	arr = np.array(im)
	source.append(arr)
# for i in range(len(displacements)):
# 	displacements[i][0] -= delta*displacements[i][1]
minx = displacements[0][0]
miny = displacements[0][1]
maxx = displacements[0][0] + source[0].shape[0]
maxy = displacements[0][1] + source[0].shape[1]
for i in range(len(displacements)):
	minx = min(minx, displacements[i][0])
	maxx = max(maxx, displacements[i][0]+source[i%len(source)].shape[0])
	miny = min(miny, displacements[i][1])
	maxy = max(maxy, displacements[i][1]+source[i%len(source)].shape[1])
for i in range(len(displacements)):
	displacements[i][0] -= minx
	displacements[i][1] -= miny
maxx -= minx
maxy -= miny
result = np.zeros((math.ceil(maxx), math.ceil(maxy),3))
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
plt.imshow(result)
plt.show()


