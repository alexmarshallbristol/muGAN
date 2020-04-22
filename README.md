# muGAN

Implementation of the SHiP muon background Generative Adversarial Network (GAN). The networks are based on those in the SHiP paper [Fast simulation of muons produced at the SHiP
experiment using Generative Adversarial Networks](https://arxiv.org/abs/1909.04451).

## How to use

First step is to import the library and initialise the muGAN class:
```
from SHiP_GAN_module import muGAN

muGAN = muGAN()
```


### Simple generation

To generate physical distributions of muon kinemtics the following command can be used:
```
muon_kinematic_vectors = muGAN.generate(size=10000)
```

The library also has a plotting function:
```
muGAN.plot_kinematics(data=muon_kinematic_vectors)
```

An example of the reults obtained with this procedure are presented in [Generated_kinematics.png](Generated_kinematics.png).