'''

	Script to be used for new combinatorial muon background analysis with most accurate GAN so far.

'''

import numpy as np
from SHiP_GAN_module import muGAN


muGAN = muGAN()

for file_id in range(0, int(1E30)):

	muon_kinematic_vectors = muGAN.generate(size=int(1E6), tuned_aux=True)

	# if file_id == 0:

	# 	muGAN.plot_kinematics(data=muon_kinematic_vectors, filename='K')

	# 	muGAN.plot_p_pt(data=muon_kinematic_vectors, filename='P')

	random_id = np.random.randint(10000000, high=99999999)

	muGAN.save_to_ROOT(data=muon_kinematic_vectors,filename='/mnt/storage/scratch/am13743/AUX_GANs_output/muons_%d.root'%random_id)
