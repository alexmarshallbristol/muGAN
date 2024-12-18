import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob

colours_raw_root = [
    [250, 242, 108],
    [249, 225, 104],
    [247, 206, 99],
    [239, 194, 94],
    [222, 188, 95],
    [206, 183, 103],
    [181, 184, 111],
    [157, 185, 120],
    [131, 184, 132],
    [108, 181, 146],
    [105, 179, 163],
    [97, 173, 176],
    [90, 166, 191],
    [81, 158, 200],
    [69, 146, 202],
    [56, 133, 207],
    [40, 121, 209],
    [27, 110, 212],
    [25, 94, 197],
    [34, 73, 162],
]

colours_raw_root = np.flip(np.divide(colours_raw_root, 256.0), axis=0)
cmp_root = mpl.colors.ListedColormap(colours_raw_root)

files = glob.glob(
    "/eos/experiment/ship/user/amarshal/FILES_FOR_CREATION_OF_ENHANCED/Muon_kinematics*.npy"
)

"""

 The following code loops through some of the GAN training files (includes charge, kinematics and auxiliary information) and does the following:

 	- plots P vs P_t histogram
 	- Cuts the number of muons allowed in each bin to 10 (bin_cut_value)
 	- Saves the auxiliary values of muons left over

 The saved array serves as the seed distribtion for the enhanced GAN generation process.

"""

""" PLAY WITH THIS VALUE """
bin_cut_value = 10  # Bear in mind this cuts the of kinematics of every file.

##############################################################################################################################


indexes = {
    "pdg": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "px": 4,
    "py": 5,
    "pz": 6,
    "xy_aux": -4,
    "z_aux": -3,
    "pxpy_aux": -2,
    "pz_aux": -1,
}


Stored_seed_kinematics = np.empty((0, 7))
Stored_seed_aux_values = np.empty((0, 4))

for file_id, file in enumerate(files[:25]):

    mu_kinematics = np.load(file)

    print(
        "Loaded file",
        file_id,
        "of",
        np.shape(files)[0],
        "it has shape",
        np.shape(mu_kinematics),
    )

    mom = np.sqrt(
        mu_kinematics[:, indexes["px"]] ** 2
        + mu_kinematics[:, indexes["py"]] ** 2
        + mu_kinematics[:, indexes["pz"]] ** 2
    )
    mom_t = np.sqrt(
        mu_kinematics[:, indexes["px"]] ** 2 + mu_kinematics[:, indexes["py"]] ** 2
    )

    hist = np.histogram2d(mom, mom_t, bins=50, range=[[0, 400], [0, 6]])

    where_over = np.where(hist[0] > bin_cut_value)

    digi = [np.digitize(mom, hist[1]), np.digitize(mom_t, hist[2])]

    to_delete = np.empty(0)
    for bin_i in range(0, np.shape(np.where(hist[0] > bin_cut_value))[1]):

        where_digi = np.where(
            (digi[0] == where_over[0][bin_i] + 1)
            & (digi[1] == where_over[1][bin_i] + 1)
        )

        to_delete = np.append(to_delete, where_digi[0][bin_cut_value:])

    mu_kinematics = np.delete(mu_kinematics, to_delete, axis=0)

    Stored_seed_kinematics = np.append(
        Stored_seed_kinematics, mu_kinematics[:, :7], axis=0
    )
    Stored_seed_aux_values = np.append(
        Stored_seed_aux_values, mu_kinematics[:, -4:], axis=0
    )

    if file_id == 0:
        mom = np.sqrt(
            Stored_seed_kinematics[:, indexes["px"]] ** 2
            + Stored_seed_kinematics[:, indexes["py"]] ** 2
            + Stored_seed_kinematics[:, indexes["pz"]] ** 2
        )
        mom_t = np.sqrt(
            Stored_seed_kinematics[:, indexes["px"]] ** 2
            + Stored_seed_kinematics[:, indexes["py"]] ** 2
        )

        plt.figure(figsize=(7, 4))
        plt.title("TESTING SCRIPT IS WORKING")
        plt.hist2d(mom, mom_t, bins=100, norm=LogNorm(), range=[[0, 400], [0, 6]])
        plt.colorbar()
        plt.savefig("Enhanced_training_distribution.png")
        plt.close("all")

    print("Saving enhanced aux values...", "Shape:", np.shape(Stored_seed_aux_values))

    np.save("Seed_auxiliary_values_for_enhanced_generation", Stored_seed_aux_values)


mom = np.sqrt(
    Stored_seed_kinematics[:, indexes["px"]] ** 2
    + Stored_seed_kinematics[:, indexes["py"]] ** 2
    + Stored_seed_kinematics[:, indexes["pz"]] ** 2
)
mom_t = np.sqrt(
    Stored_seed_kinematics[:, indexes["px"]] ** 2
    + Stored_seed_kinematics[:, indexes["py"]] ** 2
)


plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
plt.hist2d(
    mom,
    mom_t,
    bins=100,
    norm=LogNorm(),
    cmap=cmp_root,
    range=[[0, 400], [0, 7]],
    vmin=1,
)
plt.xlabel("Momentum (GeV)")
plt.ylabel("Transverse Momentum (GeV)")
plt.grid(color="k", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.text(
    0.95,
    0.95,
    "Enhanced Training Distribution",
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax.transAxes,
    fontsize=15,
)
plt.savefig("Enhanced_training_distribution.png")
plt.close("all")
