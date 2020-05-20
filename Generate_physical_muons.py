'''

	Script to be used for new combinatorial muon background analysis with most accurate GAN so far.

'''

import numpy as np
from SHiP_GAN_module import muGAN


muGAN = muGAN()

for file_id in range(0, int(1E30)):

	muon_kinematic_vectors,  muon_aux_values = muGAN.generate(size=int(1E6), tuned_aux=True)

	print(np.shape(muon_kinematic_vectors), np.shape(muon_aux_values))

	combined_array = np.concatenate((muon_kinematic_vectors, muon_aux_values),axis=1)



	random_id = np.random.randint(10000000, high=99999999)

	np.save('/mnt/storage/scratch/am13743/AUX_GANs_output/muons_and_aux_%d'%random_id,combined_array)

	# muGAN.save_to_ROOT(data=muon_kinematic_vectors,filename='/mnt/storage/scratch/am13743/AUX_GANs_output/muons_%d.root'%random_id)
