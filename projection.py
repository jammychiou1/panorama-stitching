import numpy as np
def planar_projection(xs, ys, f):
	xs_new = np.tan(xs/f)*f
	ys_new = ys / np.cos(xs/f)
	return xs_new, ys_new
def cylindrical_projection(xs, ys, f):
	xs_new = np.arctan(xs/f)*f
	ys_new = ys / (xs ** 2 + f ** 2) ** 0.5 * f
	return xs_new, ys_new
