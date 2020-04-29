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


muGAN.tune(size=int(1E6), 
	output_folder='Tuning_results',
	training_data_location='/Users/am13743/Desktop/Data_for_GAN_paper_plots/real_data.npy')


# muGAN.tune(size=int(1E6), initial_values=initial_values, 
# 	output_folder='Tuning_results',
# 	training_data_location='/mnt/storage/scratch/am13743/real_data.npy')




















