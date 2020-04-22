'''

	Script providing examples demonstrating how to use the GANs (Generative Adversarial Networks) developed 
	for the fast generation of muon kinematics generated within SHiP target. 

	Script introduces some of the functions avaliable in the module developed to clean up generation scripts.

	The GAN employed here is an updated version of those presented in the SHiP paper: https://arxiv.org/abs/1909.04451

	The new architecture introduces input auxiliary distributions. The generator has been taught to understand that these 
	auxiliary inputs correspond to information about how rare each muon is in each direction. Varying the input auxiliary 
	distributions can provide the user some control over the generated output.

	Written by A. Marshall - April 2020.

'''

import numpy as np
''' Load the muon GAN module from the SHiP_GAN_module package. '''
from SHiP_GAN_module import muGAN



########################################################################################################################################################
''' Initialise the muGAN class.'''
muGAN = muGAN()
########################################################################################################################################################








########################################################################################################################################################
''' More specific applications may want to generate an enhanced distribution. '''
''' Here is an example...'''

''' Load in your enhanced distribution (for example Oliver's distribution). '''
''' For this example I have just selected out muons with P_t > 1.5 GeV from the sample we just generated. '''
muon_kinematic_vectors_enhanced_example = np.load('/Users/am13743/Full Code GAN/Enhanced_sample.npy')
print(np.shape(muon_kinematic_vectors_enhanced_example))
muGAN.plot_p_pt(data=muon_kinematic_vectors_enhanced_example,filename='seed.png')
''' Generate based on a seed distribution. '''
muon_kinematic_vectors_enchanced = muGAN.generate_enhanced(size=np.shape(muon_kinematic_vectors_enhanced_example)[0], seed_vectors=muon_kinematic_vectors_enhanced_example, aux_multiplication_factor=1)

muGAN.plot_p_pt(data=muon_kinematic_vectors_enchanced,filename='product.png')

''' Another example of plotting kinematics of a generated vector with some other options. '''
# muGAN.plot_kinematics(data=muon_kinematic_vectors_enchanced, bins=25, log=False, filename='Generated_kinematics_ENHANCED.png', normalize_colormaps=False)
########################################################################################################################################################













