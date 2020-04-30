from __future__ import print_function
import numpy as np
import glob
import matplotlib as mpl
# mpl.use('TkAgg') 
# mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import LogNorm
import time
from scipy.stats import truncnorm


files = glob.glob('/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/transfered/mu_data*.npy')

print(np.shape(files))

# Define column names

indexes = {'weights':0, 'charge':1, 'x':2, 'y':3, 'z':4, 'px':5, 'py':6, 'pz':7, 'aux_weight':-5, 'xy_aux':-4, 'z_aux':-3, 'pxpy_aux':-2, 'pz_aux':-1 }

t0 = time.time()

for file_id, file in enumerate(files):

	print(file_id, 'of', np.shape(files))
	if file_id == 11: quit()
	# Load file
	current = np.load(file)
	# print(np.shape(current))

	# # Add 5 columns to fill with aux values
	# aux = np.ones((np.shape(current)[0],5))

	# current = np.concatenate((current, aux),axis=1)

	# current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)

	# # PROMPT AUX
	# # Set prompt_aux to -1 for prompt, and 1 for non-prompt muons
	# current[np.where((current[:,indexes['x']]==0)&(current[:,indexes['y']]==0)),indexes['prompt_aux']] = -1

	# XY AUX no prompt
	if file_id == 0: # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
		current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
		fit_to_this = current[:5000,2:8]
		nbrs_aux_weights = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)
		
	distances, indices = nbrs_aux_weights.kneighbors(current[:,2:8])
	av_distances = np.mean(distances,axis=1)
	av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances)) # safety net
	current[:,indexes['aux_weight']] = av_distances
	# Sort the array by this auxiliary value
	current = current[current[:,indexes['aux_weight']].argsort()]
	# Generate a 1 tailed Gaussian
	# half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['aux_weight']])[0],1)))
	half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['aux_weight']])[0],1))
	# Sort these values
	half_gauss = half_gauss[half_gauss[:,0].argsort()]
	# Force the auxiliary values to be these half_gauss values IN ORDER
	current[:,indexes['aux_weight']] = half_gauss[:,0]





	# # XY AUX
	# # For xy_aux, set to 0 if prompt muon
	# # current[np.where((current[:,indexes['x']]==0)&(current[:,indexes['y']]==0)),indexes['xy_aux']] = 0
	# current[np.where((current[:,indexes['x']]==0)&(current[:,indexes['y']]==0)),indexes['xy_aux']] = -5

	# if file_id == 0: # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
	# 	# only fit to non-prompt muons
	# 	fit_to_this = np.concatenate((np.expand_dims(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:5000,indexes['x']],1),np.expand_dims(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:5000,indexes['y']],1)),axis=1)
	# 	nbrs_xy = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)
		
	# distances, indices = nbrs_xy.kneighbors(np.concatenate((np.expand_dims(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,indexes['x']],1),np.expand_dims(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,indexes['y']],1)),axis=1))
	# av_distances = np.mean(distances,axis=1)
	# av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances)) # safety net
	# current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0)),indexes['xy_aux']] = av_distances
	# # Sort the array by this auxiliary value
	# current = current[current[:,indexes['xy_aux']].argsort()]
	# # Generate a 1 tailed Gaussian
	# # half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0)),indexes['xy_aux']])[1],1)))
	# half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0)),indexes['xy_aux']])[1],1))
	# # Sort these values
	# half_gauss = half_gauss[half_gauss[:,0].argsort()]
	# # Force the auxiliary values to be these half_gauss values IN ORDER - in this case only apply to the non-prompt muons
	# current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0)),indexes['xy_aux']] = np.swapaxes(half_gauss,0,1)



	# XY AUX no prompt
	if file_id == 0: # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
		current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
		fit_to_this = np.concatenate((np.expand_dims(current[:5000,indexes['x']],1),np.expand_dims(current[:5000,indexes['y']],1)),axis=1)
		nbrs_pxpy = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)
		
	distances, indices = nbrs_pxpy.kneighbors(np.concatenate((np.expand_dims(current[:,indexes['x']],1),np.expand_dims(current[:,indexes['y']],1)),axis=1))
	av_distances = np.mean(distances,axis=1)
	av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances)) # safety net
	current[:,indexes['xy_aux']] = av_distances
	# Sort the array by this auxiliary value
	current = current[current[:,indexes['xy_aux']].argsort()]
	# Generate a 1 tailed Gaussian
	# half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['xy_aux']])[0],1)))
	half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['xy_aux']])[0],1))
	# Sort these values
	half_gauss = half_gauss[half_gauss[:,0].argsort()]
	# Force the auxiliary values to be these half_gauss values IN ORDER
	current[:,indexes['xy_aux']] = half_gauss[:,0]




	# Z AUX
	if file_id == 0: # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
		current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
		nbrs_z = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(np.expand_dims(current[:5000,indexes['z']],1))
	
	distances, indices = nbrs_z.kneighbors(np.expand_dims(current[:,indexes['z']],1))
	av_distances = np.mean(distances,axis=1)
	av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances))


	current[:,indexes['z_aux']] = av_distances
	# Sort the array by this auxiliary value
	current = current[current[:,indexes['z_aux']].argsort()]
	# Generate a 1 tailed Gaussian
	# half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['z_aux']])[0],1)))
	half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['z_aux']])[0],1))
	# Sort these values
	half_gauss = half_gauss[half_gauss[:,0].argsort()]
	# Force the auxiliary values to be these half_gauss values IN ORDER
	current[:,indexes['z_aux']] = half_gauss[:,0]


	# PXPY AUX
	if file_id == 0: # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
		current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
		fit_to_this = np.concatenate((np.expand_dims(current[:5000,indexes['px']],1),np.expand_dims(current[:5000,indexes['py']],1)),axis=1)
		nbrs_pxpy = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fit_to_this)
		
	distances, indices = nbrs_pxpy.kneighbors(np.concatenate((np.expand_dims(current[:,indexes['px']],1),np.expand_dims(current[:,indexes['py']],1)),axis=1))
	av_distances = np.mean(distances,axis=1)
	av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances)) # safety net
	current[:,indexes['pxpy_aux']] = av_distances
	# Sort the array by this auxiliary value
	current = current[current[:,indexes['pxpy_aux']].argsort()]
	# Generate a 1 tailed Gaussian
	# half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['pxpy_aux']])[0],1)))
	half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['pxpy_aux']])[0],1))
	# Sort these values
	half_gauss = half_gauss[half_gauss[:,0].argsort()]
	# Force the auxiliary values to be these half_gauss values IN ORDER
	current[:,indexes['pxpy_aux']] = half_gauss[:,0]


	# PZ AUX
	if file_id == 0: # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
		current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
		nbrs_pz = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(np.expand_dims(current[:5000,indexes['pz']],1))
	
	distances, indices = nbrs_pz.kneighbors(np.expand_dims(current[:,indexes['pz']],1))
	av_distances = np.mean(distances,axis=1)
	av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances))


	current[:,indexes['pz_aux']] = av_distances
	# Sort the array by this auxiliary value
	current = current[current[:,indexes['pz_aux']].argsort()]
	# Generate a 1 tailed Gaussian
	# half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['pz_aux']])[0],1)))
	half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['pz_aux']])[0],1))
	# Sort these values
	half_gauss = half_gauss[half_gauss[:,0].argsort()]
	# Force the auxiliary values to be these half_gauss values IN ORDER
	current[:,indexes['pz_aux']] = half_gauss[:,0]

	# Shuffle at the end, for good measure

	current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)

	if file_id == 0:

		index_to_plot = [-5,-4,-3,-2,-1]
		plt.figure(figsize=(5*4,3*4))
		subplot = 0
		for ii in range(0,5):
			subplot +=1
			ii_index = index_to_plot[ii]
			plt.subplot(3,5,subplot)
			plt.hist(current[:,ii_index], bins=50, range=[-0.1,5.1])

		for ii in range(0,5):
			for jj in range(ii+1,5):
				ii_index = index_to_plot[ii]
				jj_index = index_to_plot[jj]
				subplot += 1
				plt.subplot(3,5,subplot)
				plt.hist2d(current[:,ii_index],current[:,jj_index], bins=50, range=[[-0.1,5.1],[-0.1,5.1]], norm=LogNorm())
		plt.savefig('aux_relationships.png')
		# plt.show()
		plt.close('all')

	# if file_id == 0:

	# 	# Save the 2 covariance matricies that will be used to generate aux noise during GAN training and generation.
	# 	# A perfect solution wouldnt need this and would map these aux distirbutions to the shape of uncorrelated normals.
	# 	# However that map wouldn't be perfect so maybe this solution isnt so bad.
	# 	cov_non_prompt = np.cov(np.swapaxes(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,indexes['xy_aux']:],0,1))
	# 	cov_prompt = np.cov(np.swapaxes(current[np.where((current[:,indexes['x']]==0)&(current[:,indexes['y']]==0))][:,indexes['z_aux']:],0,1))
	# 	np.save('cov_non_prompt',cov_non_prompt)
	# 	np.save('cov_prompt',cov_prompt)

	# 	# print(' ')
	# 	# gen_normal = np.random.multivariate_normal(mean=np.zeros(4), cov=cov_non_prompt, size=(np.shape(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))])[0]))
	# 	# plt.figure(figsize=(8,8))
	# 	# plt.subplot(2,2,1)
	# 	# plt.hist2d(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,-4], current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,-3], bins=50,norm=LogNorm(),range=[[-5,5],[-5,5]])
	# 	# plt.subplot(2,2,2)
	# 	# plt.hist2d(gen_normal[:,0], gen_normal[:,1], bins=50,norm=LogNorm(),range=[[-5,5],[-5,5]])

	# 	# plt.subplot(2,2,3)
	# 	# plt.hist2d(np.abs(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,-4]), np.abs(current[np.where((current[:,indexes['x']]!=0)&(current[:,indexes['y']]!=0))][:,-3]), bins=50,norm=LogNorm(),range=[[0,5],[0,5]])
	# 	# plt.subplot(2,2,4)
	# 	# plt.hist2d(np.abs(gen_normal[:,0]), np.abs(gen_normal[:,1]), bins=50,norm=LogNorm(),range=[[0,5],[0,5]])		
	# 	# plt.savefig('generate_with_cov')
	# 	# plt.show()
	# 	# plt.close('all')



	# delete prmompt

	# print(np.shape(current))
	# current = np.delete(current,[8],axis=1)
	# print(np.shape(current))
	
	t1 = time.time()
	time_n = t1-t0
	print('Time: %.1fs'%time_n)
	# quit()
	np.save('/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/aux_weight/aux_weight_mu_data_%d.npy'%file_id, current)













