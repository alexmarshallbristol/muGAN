from SHiP_GAN_module import muGAN

muGAN = muGAN()

muons = muGAN.generate(size=1e5)

muGAN.save(muons, filename="muons.root")
