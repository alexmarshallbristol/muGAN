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
seed_auxiliary_distributions = np.load('SHiP_GAN_module/data_files/Seed_auxiliary_values_for_enhanced_generation.npy')
########################################################################################################################################################
'''
	Although using these seed auxiliary distributions gets us most of the way we havent addressed the GAN underestimating the tails. 
	The tails most importantly need boosting in the P_t direction.
	The following few lines boost a set fraction of the muons. 
'''
seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)
fraction_to_boost = 0.1
cut = int(np.shape(seed_auxiliary_distributions)[0]*fraction_to_boost) 
dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(seed_auxiliary_distributions[:cut,2])))
dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(dist)))
dist += 1
dist = np.power(dist,0.5)
seed_auxiliary_distributions[:cut,2] *= dist
seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)
########################################################################################################################################################

boosted_muon_kinematic_vectors = muGAN.generate_enhanced(auxiliary_distributions=seed_auxiliary_distributions, size=int(1E5))


muGAN.plot_kinematics(data=boosted_muon_kinematic_vectors, filename='test3_EN.png', normalize_colormaps=False)
muGAN.plot_p_pt(data=boosted_muon_kinematic_vectors,filename='generated_enhanced_boosed.png')

quit()
########################################################################################################################################################
''' More specific applications may want to generate an enhanced distribution. '''
''' Here is an example...'''

''' Load in your enhanced distribution (for example Oliver's distribution). '''
''' For this example I have just selected out muons with P_t > 1.5 GeV from the sample we just generated. '''
muon_kinematic_vectors_enhanced_example = np.load('/Users/am13743/Full Code GAN/Enhanced_sample.npy')

# print(np.shape(muon_kinematic_vectors_enhanced_example))
muGAN.plot_p_pt(data=muon_kinematic_vectors_enhanced_example,filename='seed.png')

''' Generate based on a seed distribution. '''
# muon_kinematic_vectors_enchanced = muGAN.generate_enhanced_from_seed_kinematics(size=np.shape(muon_kinematic_vectors_enhanced_example)[0], 
# 									seed_vectors=muon_kinematic_vectors_enhanced_example, aux_multiplication_factor=1)





muGAN.plot_p_pt(data=muon_kinematic_vectors_enchanced,filename='product.png')

muGAN.plot_kinematics(data=muon_kinematic_vectors_enchanced, filename='product_kinematics.png', normalize_colormaps=False)


muGAN.save_to_ROOT(data=muon_kinematic_vectors_enhanced_example,filename='example.root')

########################################################################################################################################################













