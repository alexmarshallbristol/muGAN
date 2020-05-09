'''

	Script to be used for new combinatorial muon background analysis with most accurate GAN so far.

'''

import numpy as np
from SHiP_GAN_module import muGAN
import time

muGAN = muGAN()

for file_id in range(0, int(1E30)):

	t0 = time.time()

	muon_kinematic_vectors = muGAN.generate(size=int(1E5), tuned_aux=True)

	t1 = time.time()

	total_time = t1-t0

	print(total_time, np.shape(muon_kinematic_vectors))
	quit()
	# if file_id == 0:

	# 	muGAN.plot_kinematics(data=muon_kinematic_vectors, filename='K')

	# 	muGAN.plot_p_pt(data=muon_kinematic_vectors, filename='P')

	random_id = np.random.randint(10000000, high=99999999)

	muGAN.save_to_ROOT(data=muon_kinematic_vectors,filename='/mnt/storage/scratch/am13743/AUX_GANs_output/muons_%d.root'%random_id)
