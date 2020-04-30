from __future__ import print_function
import numpy as np
import glob
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt

files = glob.glob('/eos/experiment/ship/user/amarshal/THOMAS_MUONS_NPY/SPLIT*.npy')

min_max = np.load('min_max.npy')

ranges = np.empty(6)

for i, min_max_i in enumerate(min_max):
	ranges[i] = min_max_i[1] - min_max_i[0]

maxmimum_weight = 768.75

total_numbers = np.zeros((2,2)) # pos/neg, prompt/non-prompt

total_weight = 0

for file_id, file in enumerate(files):

	print(file_id, 'of', np.shape(files))

	current = np.load(file)

	# Save total weight before it is normalized

	total_weight += np.sum(current[:,0])

	# Weights 

	current[:,0] = current[:,0]/maxmimum_weight

	# Charge

	current[:,1] = current[:,1]/13

	# Save information about the sample

	total_numbers[0][0] += np.shape(np.where((current[:,2]==0)&(current[:,2]==0)&(current[:,1]==1)))[1]
	total_numbers[1][0] += np.shape(np.where((current[:,2]==0)&(current[:,2]==0)&(current[:,1]==-1)))[1]
	total_numbers[0][1] += np.shape(np.where((current[:,2]!=0)&(current[:,2]!=0)&(current[:,1]==1)))[1]
	total_numbers[1][1] += np.shape(np.where((current[:,2]!=0)&(current[:,2]!=0)&(current[:,1]==-1)))[1]
	

	# Kinematics first normalisation

	for min_max_index, kin_index in enumerate([2,3,4,5,6,7]):

		current[:,kin_index] = ((current[:,kin_index] - min_max[min_max_index][0])/ranges[min_max_index]) * 1.94 - 0.97

	# Kinematics second normalisation (x and y)

	for min_max_index, kin_index in enumerate([2,3]):

		sign = np.sign(current[np.where((current[:,2]!=0)&(current[:,3]!=0)),kin_index])

		current[np.where((current[:,2]!=0)&(current[:,3]!=0)),kin_index] = np.multiply(np.sqrt(np.abs(current[np.where((current[:,2]!=0)&(current[:,3]!=0)),kin_index])),sign)


	np.save('/eos/experiment/ship/user/amarshal/DATA_TRAINING_MUON_OPT_GAN/mu_data_%d'%file_id, current)

print(total_numbers)
print('Total muons:',int(np.sum(total_numbers)),'with a total weight of',total_weight)
Fraction_pos = float(total_numbers[0][0]+total_numbers[0][1])/float(np.sum(total_numbers))
print('Fraction positive:',Fraction_pos)
Fraction_00 = float(total_numbers[0][0]+total_numbers[1][0])/float(np.sum(total_numbers))
print('Fraction 0 0:',Fraction_00)
np.save('total_numbers',total_numbers)



















