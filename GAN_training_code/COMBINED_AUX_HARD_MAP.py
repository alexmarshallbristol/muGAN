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
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate
# from keras.layers import BatchNormalization
# from keras.models import Model
# from keras import backend as K
# from keras import objectives
# from keras.optimizers import Adam, RMSprop
# from keras.layers.advanced_activations import LeakyReLU
from scipy.stats import pearsonr

files = glob.glob('/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/mu_data_*.npy')

print(np.shape(files))

indexes = {'weights':0, 'charge':1, 'x':2, 'y':3, 'z':4, 'px':5, 'py':6, 'pz':7, 'xy_aux':-4, 'z_aux':-3, 'pxpy_aux':-2, 'pz_aux':-1 }

t0 = time.time()

for file_id, file in enumerate(files):

	# last_complete_file = -1
	last_complete_file = 17

	if file_id > last_complete_file:
		print(' ')
		print(file_id, 'of', np.shape(files))


		current = np.load(file)
		print(np.shape(current))

		current = current[:,:-1]


		# THE FOLLOWING IS A CORRECTION DUE TO DIFFERENT TARGET POSITION IN SOME FILES. 
		current = np.delete(current, np.where(current[:,4]<-0.97), axis=0)
		if file_id == (last_complete_file+1):
			min_max = np.load('min_max.npy')
			pre_value = -0.911
			post_value = (((pre_value+0.97)/1.94)*(min_max[2][1] - min_max[2][0])+ min_max[2][0])
			min_max[2][0] = post_value
			np.save('min_max2',min_max)
		current[np.where(current[:,4]<-0.911)][:,4] = current[np.where(current[:,4]<-0.911)][:,4] + (0.97-0.91)
		range_i = 1 - (-0.911)
		current[:,4] = ((current[:,4] - (-0.911))/range_i) * 2 - 1



		# XY AUX no prompt
		if file_id == (last_complete_file+1): # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
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
		half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['xy_aux']])[0],1)))
		# half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['xy_aux']])[0],1))
		# Sort these values
		half_gauss = half_gauss[half_gauss[:,0].argsort()]
		# Force the auxiliary values to be these half_gauss values IN ORDER
		current[:,indexes['xy_aux']] = half_gauss[:,0]




		# Z AUX
		if file_id == (last_complete_file+1): # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
			current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
			nbrs_z = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(np.expand_dims(current[:5000,indexes['z']],1))
		
		distances, indices = nbrs_z.kneighbors(np.expand_dims(current[:,indexes['z']],1))
		av_distances = np.mean(distances,axis=1)
		av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances))


		current[:,indexes['z_aux']] = av_distances
		# Sort the array by this auxiliary value
		current = current[current[:,indexes['z_aux']].argsort()]
		# Generate a 1 tailed Gaussian
		half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['z_aux']])[0],1)))
		# half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['z_aux']])[0],1))
		# Sort these values
		half_gauss = half_gauss[half_gauss[:,0].argsort()]
		# Force the auxiliary values to be these half_gauss values IN ORDER
		current[:,indexes['z_aux']] = half_gauss[:,0]


		# PXPY AUX
		if file_id == (last_complete_file+1): # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
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
		half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['pxpy_aux']])[0],1)))
		# half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['pxpy_aux']])[0],1))
		# Sort these values
		half_gauss = half_gauss[half_gauss[:,0].argsort()]
		# Force the auxiliary values to be these half_gauss values IN ORDER
		current[:,indexes['pxpy_aux']] = half_gauss[:,0]


		# PZ AUX
		if file_id == (last_complete_file+1): # only bother fitting this once - may change approach in the future - only needs to be approximate anyway
			current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)
			nbrs_pz = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(np.expand_dims(current[:5000,indexes['pz']],1))
		
		distances, indices = nbrs_pz.kneighbors(np.expand_dims(current[:,indexes['pz']],1))
		av_distances = np.mean(distances,axis=1)
		av_distances[np.where(av_distances==0)] = np.amin(np.nonzero(av_distances))


		current[:,indexes['pz_aux']] = av_distances
		# Sort the array by this auxiliary value
		current = current[current[:,indexes['pz_aux']].argsort()]
		# Generate a 1 tailed Gaussian
		half_gauss = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(current[:,indexes['pz_aux']])[0],1)))
		# half_gauss = truncnorm.rvs(-5, 5, size=(np.shape(current[:,indexes['pz_aux']])[0],1))
		# Sort these values
		half_gauss = half_gauss[half_gauss[:,0].argsort()]
		# Force the auxiliary values to be these half_gauss values IN ORDER
		current[:,indexes['pz_aux']] = half_gauss[:,0]

		# Shuffle at the end, for good measure

		current = np.take(current,np.random.permutation(current.shape[0]),axis=0,out=current)




		# np.save('current',current)

		# current = np.load('current.npy')



		dimensions = 4
		bins_in_gaussian= 25

		data = current[:,-dimensions:].copy()

		sample_full = data

		num_dims = dimensions

		gaussian_full = np.abs(np.random.normal(loc=0, scale=1, size=(np.shape(sample_full)[0], num_dims)))
		# cov = np.cov(np.swapaxes(sample_full,0,1))
		# gaussian_full = np.random.multivariate_normal(np.zeros(np.shape(sample_full)[1]), cov, size=np.shape(sample_full)[0])

		# gaussian_full = np.take(gaussian_full,np.random.permutation(gaussian_full.shape[0]),axis=0,out=gaussian_full)
		# sample_full = np.take(sample_full,np.random.permutation(sample_full.shape[0]),axis=0,out=sample_full)



		sample_full_pre = sample_full.copy()

		



		arrangments = np.asarray([[2,3,1,0]])



		for arrangment_index, arrangment in enumerate(arrangments):


			sample = sample_full
			gaussian = gaussian_full


			sample_pre_rearrange = sample.copy()

			# sample[:,0] = sample_pre_rearrange[:,arrangment[0]]
			# sample[:,1] = sample_pre_rearrange[:,arrangment[1]]
			# sample[:,2] = sample_pre_rearrange[:,arrangment[2]]

			# print(sample)
			# print(current[:,-dimensions:])
			# print('')
			for index_np in range(0, dimensions):
				sample[:,index_np] = sample_pre_rearrange[:,int(arrangment[index_np])]

			# print(sample)
			# print(current[:,-dimensions:])

			# plt.figure(figsize=(dimensions*4,1*4))
			# subplot = 0
			# for i in range(0, dimensions):
			# 	subplot+=1
			# 	plt.subplot(1,dimensions,subplot)
			# 	plt.hist2d(sample[:,i], current[:,-4:][:,i], bins=250,norm=LogNorm())
			# plt.subplots_adjust(hspace=0.35, wspace=0.35)
			# plt.savefig('test')
			# plt.close('all')

			# quit()

		# 	split data into 6

		# 	re arrange rows 

		# 	do the correction

		# 	revert rows back 

		# quit()


			hist_gaussian = np.histogramdd(gaussian, bins=bins_in_gaussian) # change to 50


			where_vector = np.expand_dims(np.arange(0, np.shape(sample)[0]),1)

			sample = np.concatenate((sample,where_vector), axis=1)
			gaussian = np.concatenate((gaussian,where_vector), axis=1)




			sample = sample[sample[:,0].argsort()]
			gaussian = gaussian[gaussian[:,0].argsort()]

			where_vector = np.expand_dims(np.arange(0, np.shape(sample)[0]),1)

			sample = np.concatenate((sample,where_vector), axis=1)
			gaussian = np.concatenate((gaussian,where_vector), axis=1)

			cut_index_m = 0

			

			# print('begin')

			aux_values_pre = sample.copy()



			cut_index_j = 0

			try:
				sum_hist_m = np.sum(hist_gaussian[0],axis=1)
			except:
				sum_hist_m = hist_gaussian[0]
			for loop in range(0, 10):
				try:
					sum_hist_m = np.sum(sum_hist_m,axis=1)
				except:
					break

			# for index_m, num_in_bin_m in enumerate(np.sum(np.sum(np.sum(np.sum(hist_gaussian[0],axis=1),axis=1),axis=1),axis=1)):
			for index_m, num_in_bin_m in enumerate(sum_hist_m):
				
				bin_start_m = hist_gaussian[1][0][index_m]
				bin_end_m = hist_gaussian[1][0][index_m+1]
				sample_m = sample[int(cut_index_m):int(cut_index_m+num_in_bin_m)]
				sample_m = sample_m[sample_m[:,1].argsort()]
				gaussian_m = gaussian[int(cut_index_m):int(cut_index_m+num_in_bin_m)]
				gaussian_m = gaussian_m[gaussian_m[:,1].argsort()]
				cut_index_i = 0

				try:
					sum_hist_i = np.sum(hist_gaussian[0][index_m],axis=1)
				except:
					sum_hist_i = hist_gaussian[0][index_m]
				for loop in range(0, 10):
					try:
						sum_hist_i = np.sum(sum_hist_i,axis=1)
					except:
						break

				for index_i, num_in_bin_i in enumerate(sum_hist_i):
					
					bin_start_i = hist_gaussian[1][1][index_i]
					bin_end_i = hist_gaussian[1][1][index_i+1]
					if dimensions == 2:
						if num_in_bin_i != 0: 

							sample_i = sample_m[int(cut_index_i):int(cut_index_i+num_in_bin_i)]
							gaussian_i = gaussian_m[int(cut_index_i):int(cut_index_i+num_in_bin_i)]
							sample_i = sample_i[sample_i[:,0].argsort()]
							gaussian_i = gaussian_i[gaussian_i[:,0].argsort()]
							shape = np.shape(sample_i)[0] 
							noise_m = gaussian_i[:,0]
							noise_i = gaussian_i[:,1]
							where_list = sample_i[:,-1].astype('int')
							sample[where_list,0] = noise_m
							sample[where_list,1] = noise_i
							cut_index_i += int(num_in_bin_i)   
					else:
						sample_i = sample_m[int(cut_index_i):int(cut_index_i+num_in_bin_i)]
						sample_i = sample_i[sample_i[:,2].argsort()]
						gaussian_i = gaussian_m[int(cut_index_i):int(cut_index_i+num_in_bin_i)]
						gaussian_i = gaussian_i[gaussian_i[:,2].argsort()]
						cut_index_j = 0

						try:
							sum_hist_j = np.sum(hist_gaussian[0][index_m][index_i],axis=1)
						except:
							sum_hist_j = hist_gaussian[0][index_m][index_i]
						for loop in range(0, 10):
							try:
								sum_hist_j = np.sum(sum_hist_j,axis=1)
							except:
								break

						for index_j, num_in_bin_j in enumerate(sum_hist_j):
						 
							bin_start_j = hist_gaussian[1][2][index_j]
							bin_end_j = hist_gaussian[1][2][index_j+1]
							if dimensions == 3:
								if num_in_bin_j != 0:
									sample_j = sample_i[int(cut_index_j):int(cut_index_j+num_in_bin_j)]
									gaussian_j = gaussian_i[int(cut_index_j):int(cut_index_j+num_in_bin_j)]
									sample_j = sample_j[sample_j[:,0].argsort()]
									gaussian_j = gaussian_j[gaussian_j[:,0].argsort()]
									shape = np.shape(sample_j)[0] 
									noise_m = gaussian_j[:,0]
									noise_i = gaussian_j[:,1]
									noise_j = gaussian_j[:,2]
									where_list = sample_j[:,-1].astype('int')
									sample[where_list,0] = noise_m
									sample[where_list,1] = noise_i
									sample[where_list,2] = noise_j
									cut_index_j += int(num_in_bin_j) 
							else:
								sample_j = sample_i[int(cut_index_j):int(cut_index_j+num_in_bin_j)]
								sample_j = sample_j[sample_j[:,3].argsort()]
								gaussian_j = gaussian_i[int(cut_index_j):int(cut_index_j+num_in_bin_j)]
								gaussian_j = gaussian_j[gaussian_j[:,3].argsort()]
								cut_index_k = 0

								try:
									sum_hist_k = np.sum(hist_gaussian[0][index_m][index_i][index_j],axis=1)
								except:
									sum_hist_k = hist_gaussian[0][index_m][index_i][index_j]
								for loop in range(0, 10):
									try:
										sum_hist_k = np.sum(sum_hist_k,axis=1)
									except:
										break

								for index_k, num_in_bin_k in enumerate(sum_hist_k):
								 
									bin_start_k = hist_gaussian[1][3][index_k]
									bin_end_k = hist_gaussian[1][3][index_k+1]
									if dimensions == 4:
										if num_in_bin_k != 0:
											sample_k = sample_j[int(cut_index_k):int(cut_index_k+num_in_bin_k)]
											gaussian_k = gaussian_j[int(cut_index_k):int(cut_index_k+num_in_bin_k)]
											sample_k = sample_k[sample_k[:,0].argsort()]
											gaussian_k = gaussian_k[gaussian_k[:,0].argsort()]
											shape = np.shape(sample_k)[0] 
											noise_m = gaussian_k[:,0]
											noise_i = gaussian_k[:,1]
											noise_j = gaussian_k[:,2]
											noise_k = gaussian_k[:,3]
											where_list = sample_k[:,-1].astype('int')
											sample[where_list,0] = noise_m
											sample[where_list,1] = noise_i
											sample[where_list,2] = noise_j
											sample[where_list,3] = noise_k
											cut_index_k += int(num_in_bin_k) 
									else:
										sample_k = sample_j[int(cut_index_k):int(cut_index_k+num_in_bin_k)]
										sample_k = sample_k[sample_k[:,4].argsort()]
										gaussian_k = gaussian_j[int(cut_index_k):int(cut_index_k+num_in_bin_k)]
										gaussian_k = gaussian_k[gaussian_k[:,4].argsort()]
										cut_index_l = 0
										
										try:
											sum_hist_l = np.sum(hist_gaussian[0][index_m][index_i][index_j][index_k],axis=1)
										except:
											sum_hist_l = hist_gaussian[0][index_m][index_i][index_j][index_k]
										for loop in range(0, 10):
											try:
												sum_hist_l = np.sum(sum_hist_l,axis=1)
											except:
												break

										for index_l, num_in_bin_l in enumerate(sum_hist_l):
											
											bin_start_l = hist_gaussian[1][4][index_l]
											bin_end_l = hist_gaussian[1][4][index_l+1]
											if dimensions == 5:
												if num_in_bin_l != 0:

													sample_l = sample_k[int(cut_index_l):int(cut_index_l+num_in_bin_l)]
													gaussian_l = gaussian_k[int(cut_index_l):int(cut_index_l+num_in_bin_l)]
													sample_l = sample_l[sample_l[:,0].argsort()]
													gaussian_l = gaussian_l[gaussian_l[:,0].argsort()]
													shape = np.shape(sample_l)[0] 
													noise_m = gaussian_l[:,0]
													noise_i = gaussian_l[:,1]
													noise_j = gaussian_l[:,2]
													noise_k = gaussian_l[:,3]
													noise_l = gaussian_l[:,4]
													where_list = sample_l[:,-1].astype('int')
													sample[where_list,0] = noise_m
													sample[where_list,1] = noise_i
													sample[where_list,2] = noise_j
													sample[where_list,3] = noise_k
													sample[where_list,4] = noise_l
													
													cut_index_l += int(num_in_bin_l)   
											else:
												sample_l = sample_k[int(cut_index_l):int(cut_index_l+num_in_bin_l)]
												gaussian_l = gaussian_k[int(cut_index_l):int(cut_index_l+num_in_bin_l)]
												print('Not written for more than 5 dimensions yet.')
												quit()

											if dimensions != 5:
												cut_index_l += int(num_in_bin_l)

									if dimensions != 4:
										cut_index_k += int(num_in_bin_k)
							
							if dimensions != 3:
								cut_index_j += int(num_in_bin_j)
						
					if dimensions != 2:
						cut_index_i += int(num_in_bin_i)
				
				cut_index_m += int(num_in_bin_m)
				
				
			sample, where_list = np.split(sample, [-1], axis=1)

			sample = sample[sample[:,-1].argsort()]


			sample, where_list = np.split(sample, [-1], axis=1)
				


			sample_pre_rearrange2 = sample.copy()
			for index_np in range(0, dimensions):
				sample[:,index_np] = sample_pre_rearrange2[:,int(np.where(arrangment==index_np)[0][0])]
		

		# plt.figure(figsize=(dimensions*4,1*4))
		# subplot = 0
		# for i in range(0, dimensions):
		# 	subplot+=1
		# 	plt.subplot(1,dimensions,subplot)
		# 	plt.hist2d(sample[:,i], current[:,-4:][:,i], bins=250,norm=LogNorm())
		# 	# plt.hist2d(sample_full[:,i], sample_full_pre[:,i], bins=250)
		# 	plt.title('%.5f'%pearsonr(sample[:,i], current[:,-4:][:,i])[0])
		# 	# total_pearson += pearsonr(sample_full[:,i], sample_full_pre[:,i])[0]
		# plt.subplots_adjust(hspace=0.35, wspace=0.35)
		# plt.savefig('testhere')
		# plt.close('all')


		current[:,-dimensions:] = sample

		# quit()










		
		t1 = time.time()
		time_n = t1-t0
		print('Time: %.1fs'%time_n)
		# quit()
		np.save('/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/Hard_map2/hardmap_mu_data_%d.npy'%file_id, current)













