'''
	Script providing examples demonstrating how to use the GANs (Generative Adversarial Networks) developed 
	for the fast generation of muon kinematics generated within SHiP target. 
	Script introduces some of the functions avaliable in the module developed to clean up generation scripts.
	The GAN employed here is an updated version of those presented in the SHiP paper: https://arxiv.org/abs/1909.04451
	The new architecture introduces input auxiliary distributions. The generator has been taught to understand that these 
	auxiliary inputs correspond to information about how rare each muon is in each direction. Varying the input auxiliary 
	distributions can provide the user some control over the generated output.
	Written by A. Marshall - May 2020.
'''

import numpy as np
''' Load the muon GAN module from the SHiP_GAN_module package. '''
from SHiP_GAN_module import muGAN

########################################################################################################################################################
''' Initialise the muGAN class.'''
muGAN = muGAN()
########################################################################################################################################################


muon_kinematic_vectors = np.empty((0,7))

vec_x = np.linspace(-25,25,25)
vec_y = np.linspace(-25,25,25)

pairs = 0
for i in range(0, np.shape(vec_x)[0]):
	for j in range(0, np.shape(vec_y)[0]):
		# print(vec_x[i], vec_y[j])
		muon_kinematic_vectors = np.append(muon_kinematic_vectors, [[13, vec_x[i], vec_y[j], -7084.5, 0, 0, 100]], axis=0)


vec_px = np.linspace(-7,7,25)
vec_py = np.linspace(-7,7,25)

pairs = 0
for i in range(0, np.shape(vec_px)[0]):
	for j in range(0, np.shape(vec_py)[0]):
		# print(vec_x[i], vec_y[j])
		muon_kinematic_vectors = np.append(muon_kinematic_vectors, [[13, 0, 0, -7084.5, vec_px[i], vec_py[j], 100]], axis=0)

print(np.shape(muon_kinematic_vectors))

muon_kinematic_vectors = np.append(muon_kinematic_vectors,muon_kinematic_vectors,axis=0)

muon_kinematic_vectors = np.append(muon_kinematic_vectors,muon_kinematic_vectors,axis=0)

print(np.shape(muon_kinematic_vectors))

''' Save to ROOT file. '''
muGAN.save_to_ROOT(data=muon_kinematic_vectors,filename='repeat_muons/arrays.root')



quit()


muon_kinematic_vectors = np.ones((1000,7))

muon_kinematic_vectors[:,0] = muon_kinematic_vectors[:,0] * 13

muon_kinematic_vectors[:,1] = muon_kinematic_vectors[:,1] * 1
muon_kinematic_vectors[:,2] = muon_kinematic_vectors[:,2] * 1
muon_kinematic_vectors[:,3] = muon_kinematic_vectors[:,3] * -7084.5

muon_kinematic_vectors[:,4] = muon_kinematic_vectors[:,4] * 1
muon_kinematic_vectors[:,5] = muon_kinematic_vectors[:,5] * 1
muon_kinematic_vectors[:,6] = muon_kinematic_vectors[:,6] * 100








''' Save to ROOT file. '''
muGAN.save_to_ROOT(data=muon_kinematic_vectors,filename='repeat_muons/repeat_muons_test1.root')





