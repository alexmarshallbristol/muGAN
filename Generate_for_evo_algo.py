import numpy as np

""" Load the muon GAN module from the SHiP_GAN_module package. """
from SHiP_GAN_module import muGAN

########################################################################################################################################################
""" Initialise the muGAN class."""
muGAN = muGAN()
########################################################################################################################################################


########################################################################################################################################################
""" Load kinematics of the muon distribution you want to augment """
kinematics_of_muons = muGAN.generate_SMEAR(size=5000, tuned_aux=True)[0]
"""
	Should be of the form:
		shape: [n,7], columns: [Pdg code (13 or -13), StartX, StartY, StartZ, Px, Py, Pz]
"""
muGAN.plot_p_pt(data=kinematics_of_muons, filename="p_pt_of_muons_raw.png")

muGAN.plot_kinematics(data=kinematics_of_muons, filename="kinematics_of_muons_raw.png")


generated_muon_kinematic_vectors = (
    muGAN.generate_enhanced_from_seed_kinematics_EVO_ALGO_SMEAR(
        size=5000, seed_vectors=kinematics_of_muons, aux_multiplication_factor=1
    )[0]
)
# aux_multiplication_factor variable can be used to 'boost' the kinematics if you feel that the GAN is underestimating the tails, try values like 1.1 or 1.2

muGAN.plot_p_pt(
    data=generated_muon_kinematic_vectors, filename="generated_muon_p_pt_vectors.png"
)

muGAN.plot_kinematics(
    data=generated_muon_kinematic_vectors,
    filename="generated_muon_kinematic_vectors.png",
)

""" Save to ROOT file. """
muGAN.save_to_ROOT(
    data=generated_muon_kinematic_vectors, filename="MSO_output/MSO_muons.root"
)
