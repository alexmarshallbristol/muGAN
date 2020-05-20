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


########################################################################################################################################################
''' Load seed auxiliary distribution '''
seed_auxiliary_distributions = np.load('SHiP_GAN_module/data_files/Seed_auxiliary_values_for_enhanced_generation.npy')


########################################################################################################################################################
'''
	Although using these seed auxiliary distributions gets us most of the way we havent addressed the GAN underestimating the tails. 
	The tails most importantly need boosting in the P_t direction.
	The following few lines boost a set fraction of the muons. 
	Feel free to play with this distribution.
'''
seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)
fraction_to_boost = 0.04
cut = int(np.shape(seed_auxiliary_distributions)[0]*fraction_to_boost) 
dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(seed_auxiliary_distributions[:cut,2])))
dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(dist)))
dist += 1
dist = np.power(dist,0.55)
seed_auxiliary_distributions[:cut,2] *= dist
seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)

where_array = np.where(seed_auxiliary_distributions[:,3]>3)
shape_i = int(np.shape(where_array)[1]*0.3)
where_array = where_array[0][:int(shape_i)]
seed_auxiliary_distributions[where_array,2] = seed_auxiliary_distributions[where_array,2]*1.4

where_array = np.where(seed_auxiliary_distributions[:,3]>0)
shape_i = int(np.shape(where_array)[1]*0.015)
where_array = where_array[0][:int(shape_i)]
seed_auxiliary_distributions[where_array,2] = seed_auxiliary_distributions[where_array,2]*1.3

seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)

########################################################################################################################################################

''' Generate from GAN using auxiliary seed values. '''
boosted_muon_kinematic_vectors = muGAN.generate_enhanced(auxiliary_distributions=seed_auxiliary_distributions, size=int(5E4))


''' Plot the output. '''
muGAN.plot_kinematics(data=boosted_muon_kinematic_vectors, filename='MSO_output/MSO_kinematics.png', normalize_colormaps=False)
muGAN.plot_p_pt(data=boosted_muon_kinematic_vectors,filename='MSO_output/MSO_P_PT.png')

''' Save to ROOT file. '''
muGAN.save_to_ROOT(data=boosted_muon_kinematic_vectors,filename='MSO_output/MSO_muons.root')





