from SHiP_GAN_module import muGAN

muGAN = muGAN()

for i in range(10):
    muons = muGAN.generate(size=1e6)

muGAN.save(muons, filename="muons.root")
