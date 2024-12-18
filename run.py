from SHiP_GAN_module import muGAN

muGAN = muGAN()
muon_kinematic_vectors = muGAN.generate(size=int(1E6), tuned_aux=True)

print(muon_kinematic_vectors)

muGAN.plot_kinematics(data=muon_kinematic_vectors)