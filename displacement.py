import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
import msop
import motion
def displacement(filenames,focal_length):
    displace = [np.asarray([0,0])]
    feature = []
    for i in range(len(filenames)):
        feature.append(msop.msop(filenames[i]))
    im = Image.open(filenames[0])
    arr = np.asarray(im)
    img_h = arr.shape[0]
    img_w = arr.shape[1]
    for i in range(len(filenames) - 1):
        # deltay, deltax = motion.get_model(msop.msop(filenames[i]), msop.msop(filenames[i+1]), arr.shape[1], arr.shape[0], focal_length)
        deltax, deltay = motion.get_model(feature[i],feature[i+1],img_h, img_w, focal_length)
        deltax += displace[-1][0]
        deltay += displace[-1][1]
        displace.append(np.asarray([deltax, deltay]))
    im = Image.open(filenames[-1])
    arr = np.asarray(im)
    # deltay, deltax = motion.get_model(msop.msop(filenames[-1]), msop.msop(filenames[0]), arr.shape[1], arr.shape[0],focal_length)
    deltax, deltay = motion.get_model(feature[-1], feature[0], arr.shape[1], arr.shape[0],focal_length)
    deltax += displace[-1][0]
    deltay += displace[-1][1]
    displace.append(np.asarray([deltax, deltay]))
    displace = np.asarray(displace)
    return displace
