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
				input_array[x][index] = (((input_array[x][index]+1)/1.97)*(self.min_max[index-1][1] - self.min_max[index-1][0])+ self.min_max[index-1][0])
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




	def generate(self, size):
		''' Generate muon kinematic vectors with normally distributed auxiliary values. '''
		generator = self.load_generator()

		aux_gan = np.abs(np.random.normal(0, 1, (int(size), 4)))
		charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
		gen_noise = np.random.normal(0, 1, (int(size), 100))
		images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

		print(np.amin(images[:,3]))

		images = self.post_process(images)

		print(np.amin(images[:,3]))

		print('Generated vector column names:')
		print('	Pdg, StartX, StartY, StartZ, Px, Py, Pz')
		print(' ')

		return images


	def generate_custom_aux(self, aux_gan):
		''' Generate muon kinematic vectors with custom auxiliary values. Function calculates the size variable based on the input aux distribution. '''
		if np.shape(aux_gan)[1] != 4:
			print('ERROR: Input auxiliary vectors of shape [n,4] with columns [XY_aux, Z_aux, PT_aux, PZ_aux].')
			quit()
		else:
			generator = self.load_generator()

			size = int(np.shape(aux_gan)[0])

			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(size,1),p=[1-self.Fraction_pos,self.Fraction_pos],replace=True),1)
			gen_noise = np.random.normal(0, 1, (int(size), 100))
			images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

			images = self.post_process(images)
			print('Custom aux based muon distribution generated.')
			print('Generated vector column names:')
			print('	Pdg, StartX, StartY, StartZ, Px, Py, Pz')
			print(' ')

			return images


	def generate_enhanced(self, size, seed_vectors):
		''' Generate enhanced distributions based on a seed distribution. '''

		if np.shape(seed_vectors)[1] != 7:
			print('ERROR: Input seed_vectors as vector of shape [n,7] with columns [Pdg, StartX, StartY, StartZ, Px, Py, Pz] of physical values.')
			quit()
		else:
			seed_vectors = self.pre_process(seed_vectors)

			discriminator = self.load_discriminator()

			aux_values = np.swapaxes(np.squeeze(discriminator.predict(np.expand_dims(seed_vectors,1)))[1:],0,1)

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
			print('	Pdg, StartX, StartY, StartZ, Px, Py, Pz')
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















