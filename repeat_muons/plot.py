import ROOT 
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math

# Open file 
f = ROOT.TFile.Open("ship.conical.MuonBack-TGeant4.root")
# Get ya tree
tree = f.Get("cbmsim")

# Get length of tree
N = tree.GetEntries()

print(N)


for i in range(0,N):


	tree.GetEntry(i)

	# Assign python labels to your branches
	vetoPoints = tree.vetoPoint

	for e in vetoPoints:

		print(e.GetX())
		quit()

		# probably need to do:
		# help(e)
		# my root files have methods like e.GetStartX() 
		# maybe there is a way to get leaf 

		# example of the kind of things i might save, i GetStartZ() etc wont work for you though
		save_array = np.append(save_array, [[e.GetWeight(), e.GetStartZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)


np.save('save_array.npy', save_array)


########################################################

# #maybe more pythonic to do something like this (also works). but here you cannot jump to anywhere in the file, you need to read through all of it.
# for events in tree:
# 	for e in events.branch_name: