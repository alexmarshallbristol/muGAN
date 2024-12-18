"""

	Script providing examples demonstrating how to use the GANs (Generative Adversarial Networks) developed 
	for the fast generation of muon kinematics generated within SHiP target. 

	Script introduces some of the functions avaliable in the module developed to clean up generation scripts.

	The GAN employed here is an updated version of those presented in the SHiP paper: https://arxiv.org/abs/1909.04451

	The new architecture introduces input auxiliary distributions. The generator has been taught to understand that these 
	auxiliary inputs correspond to information about how rare each muon is in each direction. Varying the input auxiliary 
	distributions can provide the user some control over the generated output.

	Written by A. Marshall - April 2020.

"""

import numpy as np

""" Load the muon GAN module from the SHiP_GAN_module package. """
from SHiP_GAN_module import muGAN


########################################################################################################################################################
""" Initialise the muGAN class."""
muGAN = muGAN()
########################################################################################################################################################


########################################################################################################################################################
""" Generate muon vectors with normally distributed auxiliary distributions... """
muon_kinematic_vectors = muGAN.generate(size=10000, tuned_aux=True)
""" The columns are as follows: Pdg, StartX, StartY, StartZ, Px, Py, Pz. """
""" These vectors can be converted to ROOT files and run in FairShip. """
""" This may be all that is needed for some applications. """

""" Something to bear inmind if using in FairShip: the Z values are based on those present in /eos/experiment/ship/data/Mbias/background-prod-2018/. """
""" Values may require shifting based on the target position in your simulation geometry. The target position in the training data was -7086. """

""" Plot kinematics of this generated vector"""
muGAN.plot_kinematics(data=muon_kinematic_vectors)
""" Plot momentum vs transverse momentum"""
muGAN.plot_p_pt(data=muon_kinematic_vectors)
########################################################################################################################################################


########################################################################################################################################################
""" More specific applications may want to generate an enhanced distribution. """
""" Here is an example..."""

""" Load in your enhanced distribution (for example Oliver's distribution). """
""" For this example I have just selected out muons with P_t > 1.5 GeV from the sample we just generated. """
muon_kinematic_vectors_enhanced_example = muon_kinematic_vectors[
    np.where(
        np.add(muon_kinematic_vectors[:, 4] ** 2, muon_kinematic_vectors[:, 5] ** 2)
        > 1.5
    )
]

""" Generate based on a seed distribution. """
muon_kinematic_vectors_enchanced = muGAN.generate_enhanced_from_seed_kinematics(
    size=1000, seed_vectors=muon_kinematic_vectors_enhanced_example
)

""" Another example of plotting kinematics of a generated vector with some other options. """
muGAN.plot_kinematics(
    data=muon_kinematic_vectors_enchanced,
    bins=25,
    log=False,
    filename="Generated_kinematics_from_seed_kinematics.png",
    normalize_colormaps=False,
)
########################################################################################################################################################


########################################################################################################################################################
""" Can play with the input auxiliary distribution. """
""" Auxiliary vectors each have 4 values: [XY_aux, Z_aux, PT_aux, PZ_aux]. """
""" Can widen the distribution on demand in whichever direction required. """
""" This example asks for muons with higher P_t. """
size = 10000
boosted_auxiliary_distribution = muGAN.generate_aux_tuned(size)
boosted_auxiliary_distribution[:, 2] *= 2
boosted_muon_kinematic_vectors = muGAN.generate_custom_aux(
    boosted_auxiliary_distribution
)
""" Plotting results... """
muGAN.plot_kinematics(
    data=boosted_muon_kinematic_vectors, filename="Generated_kinematics_custom_aux.png"
)
########################################################################################################################################################
