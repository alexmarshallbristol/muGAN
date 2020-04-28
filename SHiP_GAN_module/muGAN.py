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
import uproot

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

	def load_generator(self):
		''' Load the pre-trained generator model from the module directory. '''
		print(' ')
		print('Loading Generator...')
		generator = load_model(os.path.dirname(os.path.realpath(__file__))+'/data_files/generator.h5',custom_objects={'_loss_generator':_loss_generator})
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


	def generate_aux_tuned(self, size):
		''' Generate tuned auxiliary distribution, will parameterise later. '''

		xy_aux = np.abs(np.random.gumbel(loc=0.1,scale=0.8,size=(size)))

		####################

		z_aux = np.abs(np.random.normal(loc=0,scale=1,size=(size)))

		fraction_wider = 0.0005
		floor = int(np.floor(size*fraction_wider))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			if number > 0:
				z_aux[-number:] = np.abs(np.random.normal(loc=0,scale=1.2,size=(np.shape(z_aux[-number:]))))
				z_aux = np.take(z_aux,np.random.permutation(z_aux.shape[0]),axis=0,out=z_aux)

		fraction_gumbel = 1E-3
		floor = int(np.floor(size*fraction_gumbel))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			if number > 0:
				z_aux[-number:] = np.abs(np.random.gumbel(loc=0,scale=1,size=(np.shape(z_aux[-number:]))))
				z_aux = np.take(z_aux,np.random.permutation(z_aux.shape[0]),axis=0,out=z_aux)

		fraction_low_to_redistribute = 0.27

		floor = int(np.floor(size*fraction_low_to_redistribute))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			list_for_np_choice = np.arange(0, np.shape(z_aux)[0])
			redistribute = np.random.choice(list_for_np_choice, p=(1/(z_aux))/np.sum(1/(z_aux)), size=number, replace=False)
			if number > 0:
				z_aux_non_delete = np.delete(z_aux, redistribute, axis=0)
				list_for_np_choice = np.arange(0, np.shape(z_aux_non_delete)[0])
				z_aux[redistribute] = z_aux_non_delete[np.random.choice(list_for_np_choice, size=np.shape(z_aux[redistribute])[0], replace=False)]
		####################


		pxpy_aux = np.abs(np.random.normal(loc=0,scale=1,size=(size)))
		fraction_wider = 0.001
		floor = int(np.floor(size*fraction_wider))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			if number > 0:
				pxpy_aux[-number:] = np.abs(np.random.normal(loc=0,scale=1.2,size=(np.shape(pxpy_aux[-number:]))))
				pxpy_aux = np.take(pxpy_aux,np.random.permutation(pxpy_aux.shape[0]),axis=0,out=pxpy_aux)

		fraction_gumbel = 1E-3
		floor = int(np.floor(size*fraction_gumbel))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			if number > 0:
				pxpy_aux[-number:] = np.abs(np.random.gumbel(loc=0,scale=1,size=(np.shape(pxpy_aux[-number:]))))
				pxpy_aux = np.take(pxpy_aux,np.random.permutation(pxpy_aux.shape[0]),axis=0,out=pxpy_aux)


		fraction_low_to_redistribute = 0.12

		floor = int(np.floor(size*fraction_low_to_redistribute))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			list_for_np_choice = np.arange(0, np.shape(pxpy_aux)[0])
			redistribute = np.random.choice(list_for_np_choice, p=(1/(pxpy_aux))/np.sum(1/(pxpy_aux)), size=number, replace=False)
			if number > 0:
				z_aux_non_delete = np.delete(pxpy_aux, redistribute, axis=0)
				list_for_np_choice = np.arange(0, np.shape(z_aux_non_delete)[0])
				pxpy_aux[redistribute] = z_aux_non_delete[np.random.choice(list_for_np_choice, size=np.shape(pxpy_aux[redistribute])[0], replace=False)]


		####################

		pz_aux = np.abs(np.random.normal(loc=0,scale=1,size=(size)))
		fraction_wider = 0.003
		floor = int(np.floor(size*fraction_wider))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			if number > 0:
				pz_aux[-number:] = np.abs(np.random.normal(loc=0,scale=1.2,size=(np.shape(pz_aux[-number:]))))
				pz_aux = np.take(pz_aux,np.random.permutation(pz_aux.shape[0]),axis=0,out=pz_aux)

		fraction_low_to_redistribute = 0.2

		floor = int(np.floor(size*fraction_low_to_redistribute))
		if floor > 0:
			number = np.random.poisson(lam=floor)
			list_for_np_choice = np.arange(0, np.shape(pz_aux)[0])
			redistribute = np.random.choice(list_for_np_choice, p=(1/(pz_aux))/np.sum(1/(pz_aux)), size=number, replace=False)
			if number > 0:
				z_aux_non_delete = np.delete(pz_aux, redistribute, axis=0)
				list_for_np_choice = np.arange(0, np.shape(z_aux_non_delete)[0])
				pz_aux[redistribute] = z_aux_non_delete[np.random.choice(list_for_np_choice, size=np.shape(pz_aux[redistribute])[0], replace=False)]

		####################

		length = np.amin([np.shape(z_aux)[0], np.shape(xy_aux)[0], np.shape(pxpy_aux)[0], np.shape(pz_aux)[0]])

		tuned_aux = np.swapaxes([xy_aux[:length], z_aux[:length], pxpy_aux[:length], pz_aux[:length]],0,1)

		return tuned_aux





	def generate(self, size, tuned_aux = True):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''

		if size > 50000:
			images = self.generate_large(size, tuned_aux)   
		else:
			generator = self.load_generator()

			if tuned_aux == False:
				aux_gan = np.abs(np.random.normal(0, 1, (int(size), 4)))
			elif tuned_aux == True:
				aux_gan = self.generate_aux_tuned(int(size))
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

			images = self.post_process(images)

			print('Generated vector column names:')
			print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
			print(' ')

		return images


	def generate_large(self, size, tuned_aux = True):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''
		generator = self.load_generator()

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
				aux_gan = self.generate_aux_tuned(int(size_i))
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size_i,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size_i), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

			images = self.post_process(images)

			images_total = np.append(images_total, images, axis=0)

		if leftovers > 0:
			if tuned_aux == False:
				aux_gan = np.abs(np.random.normal(0, 1, (int(leftovers), 4)))
			elif tuned_aux == True:
				aux_gan = self.generate_aux_tuned(int(leftovers))
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


	def generate_custom_aux(self, auxiliary_distributions, size=-1):
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


			if size > 50000:
				images = self.generate_custom_aux_large(size, aux_gan)
			else:
				generator = self.load_generator()

				charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
				gen_noise = np.random.normal(0, 1, (int(size), 100))
				images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

				images = self.post_process(images)
			print('Custom aux based muon distribution generated.')
			print('Generated vector column names:')
			print(' Pdg, StartX, StartY, StartZ, Px, Py, Pz')
			print(' ')

			return images


	def generate_custom_aux_large(self, size, aux_gan):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''
		generator = self.load_generator()

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


	def generate_enhanced(self, auxiliary_distributions=np.load(os.path.dirname(os.path.realpath(__file__))+'/data_files/Seed_auxiliary_values_for_enhanced_generation.npy'), size=1):
		''' Generate muon kinematic vectors with custom auxiliary values. Function calculates the size variable based on the input aux distribution. '''

		images = self.generate_custom_aux(auxiliary_distributions=auxiliary_distributions, size=size)

		return images



	def generate_enhanced_from_seed_kinematics(self, size, seed_vectors, aux_multiplication_factor=1):
		''' Generate enhanced distributions based on a seed distribution. '''

		if np.shape(seed_vectors)[1] != 7:
			print('ERROR: Input seed_vectors as vector of shape [n,7] with columns [Pdg, StartX, StartY, StartZ, Px, Py, Pz] of physical values.')
			quit()
		else:
			seed_vectors = self.pre_process(seed_vectors)

			discriminator = self.load_discriminator()

			aux_values = np.swapaxes(np.squeeze(discriminator.predict(np.expand_dims(seed_vectors,1)))[1:],0,1)

			aux_values = aux_values * aux_multiplication_factor

			generator = self.load_generator()

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

		if np.shape(data)[1] != 7:
			print('Input a vector of shape [n,7]')
			quit()
		else:
			self.define_plotting_tools()

			print('Plotting kinematics, saving plots to:',filename)

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

		if np.shape(data)[1] != 7:
			print('Input a vector of shape [n,7]')
			quit()
		else:
			self.define_plotting_tools()

			print('Plotting kinematics, saving plots to:',filename)

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

		with uproot.recreate("example.root") as f:

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



