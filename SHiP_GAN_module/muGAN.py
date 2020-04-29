import numpy as np
from keras.models import load_model
from keras import backend as K
_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .Create_Discriminator import create_discriminator
import os


def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

class muGAN:

	def __init__(self):
		''' Constructor for this class. '''
		total_numbers = np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/total_numbers.npy')
		self.Fraction_pos = float(total_numbers[0][0]+total_numbers[0][1])/float(np.sum(total_numbers)) 
		self.min_max = np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/min_max.npy')

	def load_generator(self, generator_filename='generator.h5'):
		''' Load the pre-trained generator model from the module directory. '''
		print(' ')
		print('Loading Generator: %s ...'%generator_filename)
		generator = load_model(os.path.dirname(os.path.realpath(__file__))+'/data_files/%s'%generator_filename,custom_objects={'_loss_generator':_loss_generator})
		print('Loaded Generator.')
		print(' ')
		return generator




	def load_discriminator(self):
		''' Load the pre-trained discriminator model from its weights saved in the module directory. '''
		print(' ')
		print('Loading Discriminator...')
		discriminator = create_discriminator()
		discriminator.load_weights(os.path.dirname(os.path.realpath(__file__))+'/data_files/discriminator_weights.h5')
		print('Loaded Discriminator.')
		print(' ')
		return discriminator




	def post_process(self, input_array):
		''' Post process generated vectors into physical values. '''
		input_array[np.where(input_array[:,0]==-1),0] = -13
		input_array[np.where(input_array[:,0]!=-1),0] = 13
		for index in range(1, 3):
			for x in range(0, np.shape(input_array)[0]):
				input_array[x][index] = (((input_array[x][index]+1)/2)*(1 - (-1))+ (-1))
				input_array[x][index] = input_array[x][index] - 0
				if input_array[x][index] < 0:
					input_array[x][index] = -(input_array[x][index]**2)
				if input_array[x][index] > 0:
					input_array[x][index] = (input_array[x][index]**2)
				input_array[x][index] = input_array[x][index] + 0
				input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(self.min_max[index-1][1] - self.min_max[index-1][0])+ self.min_max[index-1][0])
		for index in [3]:
			for x in range(0, np.shape(input_array)[0]):
				input_array[x][index] = (((input_array[x][index]+1)/1.97)*(self.min_max[index-1][1] - self.min_max[index-1][0])+ self.min_max[index-1][0]) + 0.42248158 # Small correction to GAN start of target values.
		for index in range(4, 7):
			for x in range(0, np.shape(input_array)[0]):
				input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(self.min_max[index-1][1] - self.min_max[index-1][0])+ self.min_max[index-1][0])
		return input_array




	def pre_process(self, input_array):
		''' Post process generated vectors into physical values. '''
		input_array[np.where(input_array[:,0]==-13),0] = -1
		input_array[np.where(input_array[:,0]!=-13),0] = 1
		for index in [1,2,4,5,6]:
			range_i = self.min_max[index-1][1] - self.min_max[index-1][0]
			input_array[:,index] = ((input_array[:,index] - self.min_max[index-1][0])/range_i) * 1.94 - 0.97
		for index in [3]:
			range_i = self.min_max[index-1][1] - self.min_max[index-1][0]
			input_array[:,index] = ((input_array[:,index] - self.min_max[index-1][0])/range_i) * 1.97 - 1
		for index in [1,2]:
			sign = np.sign(input_array[np.where((input_array[:,1]!=0)&(input_array[:,2]!=0)),index])
			input_array[np.where((input_array[:,1]!=0)&(input_array[:,2]!=0)),index] = np.multiply(np.sqrt(np.abs(input_array[np.where((input_array[:,1]!=0)&(input_array[:,2]!=0)),index])),sign)
		return input_array


	def define_plotting_tools(self):
		''' Plotting tools for matplotlib '''
		colours_raw_root = np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/colours_raw_root.npy')
		colours_raw_root = np.flip(np.divide(colours_raw_root,256.),axis=0)
		cmp_root = mpl.colors.ListedColormap(colours_raw_root)
		cmp_root.set_under('w')
		self.cmp_root = cmp_root
		self.min_max_plot = np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/min_max_plot.npy')
		self.axis_titles = ['StartX (cm)', 'StartY (cm)', 'StartZ (cm)', 'Px (GeV)', 'Py (GeV)', 'Pz (GeV)']
		return self.min_max_plot, self.cmp_root, self.axis_titles


	def generate_aux_tuned(self, size, distribution_parameters):
		''' Generate tuned auxiliary distribution, will parameterise later. '''

		aux_values = np.abs(np.random.normal(loc=0,scale=1,size=(size,4)))

		for i in range(0, 4):

			total_covered = 0

			fraction_gumbel_A = distribution_parameters[i][0]
			loc_gumbel_A = distribution_parameters[i][1]
			scale_gumbel_A = distribution_parameters[i][2]

			number_gumble_A = int(np.floor(size*fraction_gumbel_A))

			if number_gumble_A > 0:
				aux_values[:number_gumble_A,i] = np.abs(np.random.gumbel(loc=loc_gumbel_A,scale=scale_gumbel_A,size=(np.shape(aux_values[:number_gumble_A,i]))))

			total_covered += number_gumble_A

			if total_covered < size:
				fraction_wider = distribution_parameters[i][3]
				scale_wider = distribution_parameters[i][4]

				number_wider = int(np.floor(size*fraction_wider))

				if number_wider > 0:
					aux_values[-number_wider:,i] = np.abs(np.random.normal(loc=0,scale=scale_wider,size=np.shape(aux_values[-number_wider:,i])))

					aux_values[number_gumble_A:,i] = np.take(aux_values[number_gumble_A:,i],np.random.permutation(aux_values[number_gumble_A:,i].shape[0]),axis=0,out=aux_values[number_gumble_A:,i])
				

				fraction_gumbel_B = distribution_parameters[i][5]
				number_gumble_B = int(np.floor(size*fraction_gumbel_B))

				if number_gumble_B > 0:
					aux_values[-number_gumble_B:,i] = np.abs(np.random.gumbel(loc=0,scale=1,size=np.shape(aux_values[-number_gumble_B:,i])))

					aux_values[number_gumble_A:,i] = np.take(aux_values[number_gumble_A:,i],np.random.permutation(aux_values[number_gumble_A:,i].shape[0]),axis=0,out=aux_values[number_gumble_A:,i])


			fraction_low_to_redistribute = distribution_parameters[i][6]
			number_low_to_redistribute = int(np.floor(size*fraction_low_to_redistribute))

			if number_low_to_redistribute > 0:

				list_for_np_choice = np.arange(0, np.shape(aux_values[:,i])[0])

				redistribute = np.random.choice(list_for_np_choice, p=(1/(aux_values[:,i]))/np.sum(1/(aux_values[:,i])), size=number_low_to_redistribute, replace=False)

				aux_non_delete = np.delete(aux_values[:,i], redistribute, axis=0)
				list_for_np_choice = np.arange(0, np.shape(aux_non_delete)[0])
				aux_values[:,i][redistribute] = aux_non_delete[np.random.choice(list_for_np_choice, size=np.shape(aux_values[:,i][redistribute])[0], replace=False)]

			
			aux_values[:,i] = np.take(aux_values[:,i],np.random.permutation(aux_values[:,i].shape[0]),axis=0,out=aux_values[:,i])
		


		# Moulding some correlations in the PT vs P plane:

		# from scipy.linalg import cholesky

		# r = np.array([
		# 		        [ 1,  0,  0,  0],
		# 		        [ 0,  1,  0,  0],
		# 		        [ 0,  0,  1,  -0.1],
		# 		        [ 0,  0,  -0.1,  1]
		# 		    ])
		# c = cholesky(r, lower=True)



		# where_array = np.where(aux_values[:,2]>2)

		# shape_i = int(np.shape(where_array)[1]*0.5)

		# where_array = where_array[0][:int(shape_i)]
	
		# aux_values[where_array] = np.swapaxes(np.dot(c, np.swapaxes(aux_values[where_array],0,1)),0,1)



		# r = np.array([
		# 		        [ 1,  0,  0,  0],
		# 		        [ 0,  1,  0,  0],
		# 		        [ 0,  0,  1,  -0.1],
		# 		        [ 0,  0,  -0.1,  1]
		# 		    ])
		# c = cholesky(r, lower=True)

		# where_array = np.where(aux_values[:,3]>2)

		# shape_i = int(np.shape(where_array)[1]*0.8)

		# where_array = where_array[0][:int(shape_i)]
	
		# aux_values[where_array] = np.swapaxes(np.dot(c, np.swapaxes(aux_values[where_array],0,1)),0,1)


		where_array = np.where(aux_values[:,3]>3)

		shape_i = int(np.shape(where_array)[1]*0.8)

		where_array = where_array[0][:int(shape_i)]


		# print(aux_values[where_array,2])
		aux_values[where_array,2] = aux_values[where_array,2]*0.7
		# print(aux_values[where_array,2])


		return aux_values





	def generate(self, size, tuned_aux = True, generator_filename='generator.h5'):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''

		if size > 50000:
			images = self.generate_large(size, tuned_aux, generator_filename=generator_filename)   
		else:
			generator = self.load_generator(generator_filename=generator_filename)

			if tuned_aux == False:
				aux_gan = np.abs(np.random.normal(0, 1, (int(size), 4)))
			elif tuned_aux == True:
				aux_gan = self.generate_aux_tuned(int(size), np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/tuned_aux_parameters.npy'))
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

			images = self.post_process(images)

			print('Generated vector column names:')
			print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
			print(' ')

		return images


	def generate_large(self, size, tuned_aux, generator_filename):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''
		generator = self.load_generator(generator_filename=generator_filename)

		images_total = np.empty((0,7))

		size_i = 50000 

		print(' ')
		print(' ')
		print('Large number to generate, generating in batches of',size_i,'.')

		iterations = int(np.floor(size/size_i))

		leftovers = size - (size_i*iterations)

		for i in range(0, iterations):

			print('Generated',np.shape(images_total)[0],'muons so far...')

			if tuned_aux == False:
				aux_gan = np.abs(np.random.normal(0, 1, (int(size_i), 4)))
			elif tuned_aux == True:
				aux_gan = self.generate_aux_tuned(int(size_i), np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/tuned_aux_parameters.npy'))

			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size_i,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size_i), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

			images = self.post_process(images)

			images_total = np.append(images_total, images, axis=0)

		if leftovers > 0:
			if tuned_aux == False:
				aux_gan = np.abs(np.random.normal(0, 1, (int(leftovers), 4)))
			elif tuned_aux == True:
				aux_gan = self.generate_aux_tuned(int(leftovers), np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/tuned_aux_parameters.npy'))
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(leftovers,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(leftovers), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

			images = self.post_process(images)

			images_total = np.append(images_total, images, axis=0)

		print('Generated',np.shape(images_total)[0],'muons.')
		print('Generated vector column names:')
		print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
		print(' ')

		return images_total


	def generate_custom_aux(self, auxiliary_distributions, size=-1, generator_filename='generator.h5'):
		''' Generate muon kinematic vectors with custom auxiliary values. Function calculates the size variable based on the input aux distribution. '''
		aux_gan = auxiliary_distributions
		if np.shape(aux_gan)[1] != 4:
			print('ERROR: Input auxiliary vectors of shape [n,4] with columns [XY_aux, Z_aux, PT_aux, PZ_aux].')
			quit()
		else:
			if size == -1:
				size = int(np.shape(aux_gan)[0])
			else:
				aux_gan = np.take(aux_gan,np.random.permutation(aux_gan.shape[0]),axis=0,out=aux_gan)
				aux_gan = aux_gan[:size]

			print('Producing enhanced distribution of ',size,'muons, based on',np.shape(aux_gan)[0],'seed muons.')

			# Pick random choices WITH REPLACEMENT from the seed auxiliary distribution
			# Replication shouldn't matter, or effects will be small, as most of the variation will come in from the gen_noise vector.
			aux_gan = aux_gan[np.random.choice(np.arange(np.shape(aux_gan)[0]),size=(size),replace=True)]


			if size > 50000:
				images = self.generate_custom_aux_large(size, aux_gan)
			else:
				generator = self.load_generator(generator_filename=generator_filename)

				charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
				gen_noise = np.random.normal(0, 1, (int(size), 100))
				images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

				images = self.post_process(images)
			print('Custom aux based muon distribution generated.')
			print('Generated vector column names:')
			print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
			print(' ')

			return images


	def generate_custom_aux_large(self, size, aux_gan, generator_filename='generator.h5'):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''
		generator = self.load_generator(generator_filename=generator_filename)

		images_total = np.empty((0,7))

		size_i = 50000 

		print(' ')
		print(' ')
		print('Large number to generate, generating in batches of',size_i,'.')

		iterations = int(np.floor(size/size_i))

		leftovers = size - (size_i*iterations)

		indexes = [0, size_i]

		for i in range(0, iterations):

			print('Generated',np.shape(images_total)[0],'muons so far...')

			aux_gan_i = aux_gan[indexes[0]:indexes[1]]

			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size_i,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size_i), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan_i,1), charge_gan]))

			images = self.post_process(images)

			images_total = np.append(images_total, images, axis=0)

			indexes[0] += size_i
			indexes[1] += size_i

		if leftovers > 0:
			aux_gan_i = aux_gan[-leftovers:]
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(leftovers,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(leftovers), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan_i,1), charge_gan]))

			images = self.post_process(images)

			images_total = np.append(images_total, images, axis=0)

		print('Generated',np.shape(images_total)[0],'muons.')
		print('Generated vector column names:')
		print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
		print(' ')

		return images_total


	def generate_enhanced(self, auxiliary_distributions=np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/Seed_auxiliary_values_for_enhanced_generation.npy'), size=-1, generator_filename='generator.h5'):
		''' Generate muon kinematic vectors with custom auxiliary values. Function calculates the size variable based on the input aux distribution. '''
		if size == -1:
			size = np.shape(auxiliary_distributions)[0]
			
		images = self.generate_custom_aux(auxiliary_distributions=auxiliary_distributions, size=size)

		return images



	def generate_enhanced_from_seed_kinematics(self, size, seed_vectors, aux_multiplication_factor=1, generator_filename='generator.h5'):
		''' Generate enhanced distributions based on a seed distribution. '''

		if np.shape(seed_vectors)[1] != 7:
			print('ERROR: Input seed_vectors as vector of shape [n,7] with columns [Pdg, StartX, StartY, StartZ, Px, Py, Pz] of physical values.')
			quit()
		else:
			seed_vectors = self.pre_process(seed_vectors)

			discriminator = self.load_discriminator()

			aux_values = np.swapaxes(np.squeeze(discriminator.predict(np.expand_dims(seed_vectors,1)))[1:],0,1)

			aux_values = aux_values * aux_multiplication_factor

			generator = self.load_generator(generator_filename=generator_filename)

			print('Producing enhanced distribution of ',size,'muons, based on',np.shape(aux_values)[0],'seed muons.')

			# Pick random choices WITH REPLACEMENT from the seed auxiliary distribution
			# Replication shouldn't matter, or effects will be small, as most of the variation will come in from the gen_noise vector.
			aux_values = aux_values[np.random.choice(np.arange(np.shape(aux_values)[0]),size=(size),replace=True)]

			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size), 100))

			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_values,1), charge_gan]))

			images = self.post_process(images)

			print('Enhanced distribution generated.')
			print('Generated vector column names:')
			print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
			print(' ')

			return images

	def plot_kinematics(self, data, filename='Generated_kinematics.png',log=True, bins=100, normalize_colormaps=True):
		''' Plot the kinematics of an input vector. The input is assumed to be of columns [Pdg, StartX, StartY, StartZ, Px, Py, Pz] in an [n,7] shape. '''

		if np.shape(data)[1] not in [6, 7]:
			print('Input a vector of shape [n,6] or [n,7]')
			quit()
		else:
			self.define_plotting_tools()

			print('Plotting kinematics, saving plots to:',filename)

			if np.shape(data)[1] == 7:
				data = data[:,1:] # Cut off Pdg information for plotting

			if normalize_colormaps == True:
				max_bin = 0
				for i in range(0, 6):
					for j in range(i+1, 6):
						hist = np.histogram2d(data[:,i], data[:,j], bins=bins, range=[[self.min_max_plot[i][0], self.min_max_plot[i][1]],[self.min_max_plot[j][0],self.min_max_plot[j][1]]])
						if np.amax(hist[0])>max_bin:
							max_bin = np.amax(hist[0])

			plt.figure(figsize=(22, 12))
			subplot=0
			for i in range(0, 6):
				for j in range(i+1, 6):
					subplot += 1
					plt.subplot(3,5,subplot)
					if log == True:
						if normalize_colormaps == True:
							plt.hist2d(data[:,i], data[:,j], bins=bins, norm=LogNorm(), cmap=self.cmp_root, range=[[self.min_max_plot[i][0], self.min_max_plot[i][1]],[self.min_max_plot[j][0],self.min_max_plot[j][1]]], vmin=1, vmax=max_bin)
						else:
							plt.hist2d(data[:,i], data[:,j], bins=bins, norm=LogNorm(), cmap=self.cmp_root, range=[[self.min_max_plot[i][0], self.min_max_plot[i][1]],[self.min_max_plot[j][0],self.min_max_plot[j][1]]],vmin=1)
					else:
						if normalize_colormaps == True:
							plt.hist2d(data[:,i], data[:,j], bins=bins, cmap=self.cmp_root, range=[[self.min_max_plot[i][0], self.min_max_plot[i][1]],[self.min_max_plot[j][0],self.min_max_plot[j][1]]], vmin=1, vmax=max_bin)
						else:
							plt.hist2d(data[:,i], data[:,j], bins=bins, cmap=self.cmp_root, range=[[self.min_max_plot[i][0], self.min_max_plot[i][1]],[self.min_max_plot[j][0],self.min_max_plot[j][1]]],vmin=1)
					plt.xlabel(self.axis_titles[i])
					plt.ylabel(self.axis_titles[j])
					plt.grid(color='k',linestyle='--',alpha=0.4)
			plt.subplots_adjust(wspace=0.3, hspace=0.3)
			plt.tight_layout()
			plt.savefig(filename)
			plt.close('all')



	def plot_p_pt(self, data, filename='Generated_p_pt.png',log=True, bins=100):
		''' Plot the kinematics of an input vector. The input is assumed to be of columns [Pdg, StartX, StartY, StartZ, Px, Py, Pz] in an [n,7] shape. '''

		if np.shape(data)[1] not in [6, 7]:
			print('Input a vector of shape [n,6] or [n,7]')
			quit()
		else:
			self.define_plotting_tools()

			print('Plotting kinematics, saving plots to:',filename)

			if np.shape(data)[1] == 7:
				data = data[:,1:] # Cut off Pdg information for plotting

			mom = np.sqrt(data[:,-1]**2+data[:,-2]**2+data[:,-3]**2)
			mom_t = np.sqrt(data[:,-2]**2+data[:,-3]**2)

			plt.figure(figsize=(6, 4))
			if log == True:
				plt.hist2d(mom, mom_t, bins=bins, norm=LogNorm(), cmap=self.cmp_root, range=[[0,400],[0,7]],vmin=1)
			else:
				plt.hist2d(mom, mom_t, bins=bins, cmap=self.cmp_root, range=[[0,400],[0,7]],vmin=1)
			plt.xlabel('Momentum (GeV)')
			plt.ylabel('Transverse Momentum (GeV)')
			plt.grid(color='k',linestyle='--',alpha=0.4)
			plt.tight_layout()
			plt.savefig(filename)
			plt.close('all')



	def save_to_ROOT(self, data, filename = 'muons.root'):
		'''  Use uproot to save a generated array to a ROOT file that is compalible with MuonBackGenerator.cxx from FairShip'''
		import uproot

		shape = np.shape(data)[0]

		data[:,3] += 2084.5 # Shift target to 50m. In accordance with primGen.SetTarget(ship_geo.target.z0+50*u.m,0.) in run_simScript.py
							# The start of target in the GAN training data is -7084.5.

		dtype = '>f4'

		Event_ID = uproot.newbranch(dtype)
		ID = uproot.newbranch(dtype)
		Parent_ID = uproot.newbranch(dtype)
		Pythia_ID = uproot.newbranch(dtype)
		ECut = uproot.newbranch(dtype)
		W = uproot.newbranch(dtype)
		X = uproot.newbranch(dtype)
		Y = uproot.newbranch(dtype)
		Z = uproot.newbranch(dtype)
		PX = uproot.newbranch(dtype)
		PY = uproot.newbranch(dtype)
		PZ = uproot.newbranch(dtype)
		Release_Time = uproot.newbranch(dtype)
		Mother_ID = uproot.newbranch(dtype)
		Process_ID = uproot.newbranch(dtype)

		branchdict = {"event_id": Event_ID, "id": ID, "parentid": Parent_ID, "pythiaid": Pythia_ID, "ecut": ECut , "w": W,
		 "x": X, "y": Y, "z": Z, "px": PX, "py": PY, "pz": PZ, "release_time": Release_Time, "mother_id": Mother_ID, "process_id": Process_ID}

		tree = uproot.newtree(branchdict, title="pythia8-Geant4")

		with uproot.recreate(filename) as f:

			f["pythia8-Geant4"] = tree

			f["pythia8-Geant4"].extend({"event_id": np.ones(shape).astype(np.float64), "id": np.array(np.ones(shape)*13).astype(np.float64), "parentid": np.zeros(shape).astype(np.float64),
				"pythiaid": data[:,0].astype(np.float64), "ecut": np.array(np.ones(shape)*0.00001).astype(np.float64), "w": np.ones(shape).astype(np.float64), "x": np.array(data[:,1]*0.01).astype(np.float64),
				"y": np.array(data[:,2]*0.01).astype(np.float64), "z": np.array(data[:,3]*0.01).astype(np.float64), "px": data[:,4].astype(np.float64), "py": data[:,5].astype(np.float64),
				"pz": data[:,6].astype(np.float64), "release_time": np.zeros(shape).astype(np.float64), "mother_id": np.array(np.ones(shape)*99).astype(np.float64), "process_id": np.array(np.ones(shape)*99).astype(np.float64)})
			# Not clear if all the datatype formatting is needed. Can be fiddly with ROOT datatypes. This works so I left it.


			# Add buffer event at the end. This will not be read into simulation.
			f["pythia8-Geant4"].extend({"event_id": [0], "id": [0], "parentid": [0],
				"pythiaid": [0], "ecut": [0], "w": [0], "x": [0],
				"y": [0], "z": [0], "px": [0], "py": [0],
				"pz": [0], "release_time": [0], "mother_id": [0], "process_id": [0]})

		print(' ')
		print(' ')
		print('Saved',shape,'muons to',filename,'.')
		print('run_simScript.py must be run with the option: -n',shape,'(or lower)')
		print(' ')
		print(' ')

	def compare_generators(self, size=10000, size_enhanced=10000, generator_list=['generator.h5'], output_folder='',training_data_location='/Users/am13743/Desktop/Data_for_GAN_paper_plots/real_data.npy'):
		''' Generate muon kinematic vectors with custom auxiliary values. Function calculates the size variable based on the input aux distribution. '''
		# auxiliary_distributions=np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/Seed_auxiliary_values_for_enhanced_generation.npy')
		self.define_plotting_tools()

		X_train = np.load(training_data_location)
		if size > np.shape(X_train)[0]:
			print('Qutting... size too big, max size:',np.shape(X_train))
			quit()

		random_indices = np.random.choice(np.shape(X_train)[0], size=size, replace=False)
		X_train = X_train[random_indices]

		number_of_generators = len(generator_list)

		images_list = np.empty((number_of_generators,size,6))
		for index, generator in enumerate(generator_list):
			images = self.generate(size, tuned_aux = False, generator_filename=generator)[:,1:]
			images_list[index] = images

		

		# Values

		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 6):
			subplot += 1

			plt.subplot(2,3,subplot)

			plt.hist(X_train[:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]], label='Train',histtype='step')

			for generator_index in np.arange(0, number_of_generators):

				label = generator_list[generator_index]
				plt.hist(images_list[generator_index,:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]],histtype='step',label=label)

			plt.xlabel(self.axis_titles[i])
			if self.axis_titles[i] == 'StartZ (cm)': plt.legend(fontsize=8)

		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s/Values.png'%(output_folder),bbox_inches='tight')
		plt.close('all')



		# Values log

		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 6):
			subplot += 1

			plt.subplot(2,3,subplot)

			plt.hist(X_train[:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]], label='Train',histtype='step')

			for generator_index in np.arange(0, number_of_generators):

				label = generator_list[generator_index]
				plt.hist(images_list[generator_index,:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]],histtype='step',label=label)

			plt.yscale('log')

			plt.xlabel(self.axis_titles[i])
			if self.axis_titles[i] == 'StartZ (cm)': plt.legend(fontsize=8)

		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s/Values_log.png'%(output_folder),bbox_inches='tight')
		plt.close('all')

		# Correlations

		self.plot_kinematics(data=X_train, filename='%s/Correlations_Train.png'%output_folder)
		self.plot_p_pt(data=X_train, filename='%s/P_PT_Train.png'%output_folder)
		for generator_index, generator in enumerate(generator_list):
			label = generator_list[generator_index][:-3]
			self.plot_kinematics(data=images_list[generator_index], filename='%s/Correlations_%s.png'%(output_folder,label))
			self.plot_p_pt(data=images_list[generator_index], filename='%s/P_PT_%s.png'%(output_folder,label))


		# Enhanced distributions

		seed_auxiliary_distributions = np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/Seed_auxiliary_values_for_enhanced_generation.npy')
		seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)
		fraction_to_boost = 0.125
		cut = int(np.shape(seed_auxiliary_distributions)[0]*fraction_to_boost) 
		dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(seed_auxiliary_distributions[:cut,2])))
		dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(dist)))
		dist += 1
		dist = np.power(dist,0.6)
		seed_auxiliary_distributions[:cut,2] *= dist
		seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,np.random.permutation(seed_auxiliary_distributions.shape[0]),axis=0,out=seed_auxiliary_distributions)

		images_enhanced_list = np.empty((number_of_generators,size_enhanced,6))
		for index, generator in enumerate(generator_list):
			images = self.generate_enhanced(auxiliary_distributions=seed_auxiliary_distributions, size=size_enhanced)[:,1:]
			images_enhanced_list[index] = images

		for generator_index, generator in enumerate(generator_list):
			label = generator_list[generator_index][:-3]
			self.plot_kinematics(data=images_enhanced_list[generator_index], filename='%s/ENH_Correlations_%s.png'%(output_folder,label))
			self.plot_p_pt(data=images_enhanced_list[generator_index], filename='%s/ENH_P_PT_%s.png'%(output_folder,label))










	def tune(self, size=10000, initial_values=np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/tuned_aux_parameters.npy'), output_folder='Tuning_results', training_data_location='/mnt/storage/scratch/am13743/real_data.npy'):
		''' Generate tuned auxiliary distribution, will parameterise later. '''

		self.define_plotting_tools()

		X_train = np.load(training_data_location)
		if size > np.shape(X_train)[0]:
			print('Qutting... size too big, max size:',np.shape(X_train))
			quit()

		random_indices = np.random.choice(np.shape(X_train)[0], size=size, replace=False)
		X_train = X_train[random_indices]


		self.plot_kinematics(data=X_train, filename='%s/Correlations_Train.png'%(output_folder))
		self.plot_p_pt(data=X_train, filename='%s/P_PT_Train.png'%(output_folder))


	

		# loop smartly over parameters here.
		distribution_parameters = initial_values

		label ='initial_values'

		auxiliary_distributions = self.generate_aux_tuned(size, distribution_parameters)

		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 4):
			for j in range(i+1, 4):
				subplot += 1
				plt.subplot(2,3,subplot)
				plt.hist2d(auxiliary_distributions[:,i], auxiliary_distributions[:,j], bins=100, norm=LogNorm(), cmap=self.cmp_root, range=[[0,8],[0,8]])
				plt.grid(color='k',linestyle='--',alpha=0.4)
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.tight_layout()
		plt.savefig('%s/Aux_values_%s.png'%(output_folder,label))
		plt.close('all')


		images = self.generate_custom_aux(auxiliary_distributions, size=size, generator_filename='generator.h5')[:,1:]





		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 6):
			subplot += 1
			plt.subplot(2,3,subplot)
			plt.hist(X_train[:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]], label='Train',histtype='step')
			# label = 'Gen'
			plt.hist(images[:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]],histtype='step',label=label)
			plt.xlabel(self.axis_titles[i])
			if self.axis_titles[i] == 'StartZ (cm)': plt.legend(fontsize=8)
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s/Values_%s.png'%(output_folder,label),bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 6):
			subplot += 1
			plt.subplot(2,3,subplot)
			plt.hist(X_train[:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]], label='Train',histtype='step')
			# label = 'Gen'
			plt.hist(images[:,i], bins=51,range=[self.min_max_plot[i][0], self.min_max_plot[i][1]],histtype='step',label=label)
			plt.xlabel(self.axis_titles[i])
			plt.yscale('log')
			if self.axis_titles[i] == 'StartZ (cm)': plt.legend(fontsize=8)
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s/Values_log_%s.png'%(output_folder,label),bbox_inches='tight')
		plt.close('all')



		self.plot_kinematics(data=images, filename='%s/Correlations_%s.png'%(output_folder,label))
		self.plot_p_pt(data=images, filename='%s/P_PT_%s.png'%(output_folder,label))










