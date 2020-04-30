from __future__ import print_function
import numpy as np
import glob

files = glob.glob('/eos/experiment/ship/user/amarshal/THOMAS_MUONS_NPY/SPLIT*.npy')

min_max = np.empty((6,2))

for file_id,file in enumerate(files):
	print(file_id, 'of',np.shape(files))

	current = np.load(file)

	for min_max_id, index in enumerate([2,3,4,5,6,7]):
		if file_id == 0:
			min_max[min_max_id][0] = np.amin(current[:,index])
			min_max[min_max_id][1] = np.amax(current[:,index])
		else:
			if np.amin(current[:,index]) < min_max[min_max_id][0]:
				min_max[min_max_id][0] = np.amin(current[:,index])
			if np.amax(current[:,index]) > min_max[min_max_id][1]:
				min_max[min_max_id][1] = np.amax(current[:,index])


# Want symmetrical normalisation!

min_xy = np.amin([min_max[0][0], min_max[1][0]])
max_xy = np.amax([min_max[0][1], min_max[1][1]])
value_xy = np.amax(np.abs([min_xy, max_xy]))
min_max[0][0] = min_max[1][0] = -value_xy
min_max[0][1] = min_max[1][1] = value_xy


min_pxy = np.amin([min_max[3][0], min_max[4][0]])
max_pxy = np.amax([min_max[3][1], min_max[4][1]])
value_pxy = np.amax(np.abs([min_pxy, max_pxy]))
min_max[3][0] = min_max[4][0] = -value_pxy
min_max[3][1] = min_max[4][1] = value_pxy

np.save('min_max',min_max)
print(min_max)














