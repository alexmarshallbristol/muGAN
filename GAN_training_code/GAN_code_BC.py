import numpy as np

from keras.layers import Input, Flatten, Dense, Reshape, Dropout, BatchNormalization, Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras import backend as K
_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 

import math

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import truncnorm

import glob

import time

import shutil

colours_raw_root = [[250,242,108],
					[249,225,104],
					[247,206,99],
					[239,194,94],
					[222,188,95],
					[206,183,103],
					[181,184,111],
					[157,185,120],
					[131,184,132],
					[108,181,146],
					[105,179,163],
					[97,173,176],
					[90,166,191],
					[81,158,200],
					[69,146,202],
					[56,133,207],
					[40,121,209],
					[27,110,212],
					[25,94,197],
					[34,73,162]]

colours_raw_root = np.flip(np.divide(colours_raw_root,256.),axis=0)
cmp_root = mpl.colors.ListedColormap(colours_raw_root)


def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val


##############################################################################################################

# G_architecture = [512,512]
# D_architecture = [256,512]
# D_architecture_prompt = D_architecture_xy = D_architecture_z = D_architecture_pxpy = D_architecture_pz = [10, 10]

G_architecture = [2000,3000]
D_architecture = [1500,2000]
D_architecture_xy = D_architecture_z = D_architecture_pxpy = D_architecture_pz = [32, 64]


weight_of_vanilla_loss = 10
# weight_of_vanilla_loss = 30


working_directory = '/mnt/storage/scratch/am13743/MUON_SHIELD_OPT_GAN/'
training_directory = '/mnt/storage/scratch/am13743/DATA_TRAINING_MUON_OPT_GAN_hardmap2/'
training_name = 'hardmap_mu_data_*.npy'
saving_directory = 'test_0'
total_numbers = np.load('total_numbers.npy')

# working_directory = '/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/vae_stuff/'
# training_directory = '/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/vae_stuff/'
# training_name = 'uncorr_sing_aux_weight_mu_data_*.npy'
# saving_directory = 'test'
# total_numbers = np.load('/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/total_numbers.npy')


Fraction_pos = float(total_numbers[0][0]+total_numbers[0][1])/float(np.sum(total_numbers))


epochs = int(1E30)
batch = 50
# save_interval = 25000
save_interval = 500
enhanced_GAN = True
generator_noise_approach = 2

##############################################################################################################

print(' ')
print('Initializing networks...')
print(' ')



# Define optimizers ...

optimizerG = Adam(lr=0.0004, beta_1=0.5, decay=0, amsgrad=True)
optimizerD = Adam(lr=0.0004, beta_1=0.5, decay=0, amsgrad=True)



##############################################################################################################
# Build Generative model ...
print(' ')
print('Generator...')
print(' ')

input_noise = Input(shape=(1,100))
auxiliary_inputs = Input(shape=(1,4))
charge_input = Input(shape=(1,1))

initial_state = Concatenate()([input_noise,auxiliary_inputs,charge_input])

H = Dense(int(G_architecture[0]))(initial_state)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

for layer in G_architecture[1:]:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)

H = Dense(6, activation='tanh')(H)
g_output = Reshape((1,6))(H)

g_output = Concatenate()([charge_input, g_output])

generator = Model(inputs=[input_noise,auxiliary_inputs,charge_input], outputs=[g_output])

generator.compile(loss=_loss_generator, optimizer=optimizerG)
generator.summary()
# quit()
##############################################################################################################



print(' ')
print('Discriminator...')
print(' ')

##############################################################################################################
# Build Discriminator xy model ...

d_input = Input(shape=(1,7))

H_x = split_tensor(1, d_input)
H_y = split_tensor(2, d_input)
H_xy = Concatenate(axis=-1)([H_x, H_y])
H_xy = Reshape((1,2))(H_xy)

H = Flatten()(H_xy)

for layer in D_architecture_xy:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output_aux = Dense(1, activation='relu')(H)

discriminator_aux_xy = Model(d_input, [d_output_aux])

discriminator_aux_xy.compile(loss=[mean_squared_error],optimizer=optimizerD)
##############################################################################################################

##############################################################################################################
# Build Discriminator z model ...

d_input = Input(shape=(1,7))

H_z = split_tensor(3, d_input)
H_z = Reshape((1,1))(H_z)

H = Flatten()(H_z)

for layer in D_architecture_z:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output_aux = Dense(1, activation='relu')(H)

discriminator_aux_z = Model(d_input, [d_output_aux])

discriminator_aux_z.compile(loss=[mean_squared_error],optimizer=optimizerD)
##############################################################################################################


##############################################################################################################
# Build Discriminator pxpy model ...

d_input = Input(shape=(1,7))

H_px = split_tensor(4, d_input)
H_py = split_tensor(5, d_input)
H_pxpy = Concatenate(axis=-1)([H_px, H_py])
H_pxpy = Reshape((1,2))(H_pxpy)

H = Flatten()(H_pxpy)

for layer in D_architecture_pxpy:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output_aux = Dense(1, activation='relu')(H)

discriminator_aux_pxpy = Model(d_input, [d_output_aux])

discriminator_aux_pxpy.compile(loss=[mean_squared_error],optimizer=optimizerD)
##############################################################################################################

##############################################################################################################
# Build Discriminator pz model ...

d_input = Input(shape=(1,7))

H_pz = split_tensor(6, d_input)
H_pz = Reshape((1,1))(H_pz)

H = Flatten()(H_pz)

for layer in D_architecture_pz:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output_aux = Dense(1, activation='relu')(H)

discriminator_aux_pz = Model(d_input, [d_output_aux])

discriminator_aux_pz.compile(loss=[mean_squared_error],optimizer=optimizerD)
##############################################################################################################


##############################################################################################################
# Build Discriminator model ...

d_input = Input(shape=(1,7))

H = Flatten()(d_input)

for layer in D_architecture:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)
d_output = Dense(1, activation='sigmoid')(H)

make_trainable(discriminator_aux_xy, False)
make_trainable(discriminator_aux_z, False)
make_trainable(discriminator_aux_pxpy, False)
make_trainable(discriminator_aux_pz, False)

d_output_aux_i = discriminator_aux_xy(d_input)
d_output_aux_j = discriminator_aux_z(d_input)
d_output_aux_k = discriminator_aux_pxpy(d_input)
d_output_aux_l = discriminator_aux_pz(d_input)

discriminator = Model(d_input, [d_output, d_output_aux_i, d_output_aux_j, d_output_aux_k, d_output_aux_l])

discriminator.compile(loss=['binary_crossentropy',mean_squared_error,mean_squared_error,mean_squared_error,mean_squared_error],optimizer=optimizerD,loss_weights=[1,0,0,0,0])
##############################################################################################################


##############################################################################################################
# Build stacked GAN model ...
print(' ')
print('Stacking...')
print(' ')
make_trainable(discriminator, False)

input_noise = Input(shape=(1,100))
auxiliary_inputs = Input(shape=(1,4))
charge_input = Input(shape=(1,1))

H = generator([input_noise, auxiliary_inputs, charge_input])

gan_output, gan_output_aux_i, gan_output_aux_j, gan_output_aux_k, gan_output_aux_l = discriminator(H)

GAN_stacked = Model(inputs=[input_noise,auxiliary_inputs,charge_input], outputs=[gan_output, gan_output_aux_i, gan_output_aux_j, gan_output_aux_k, gan_output_aux_l])
GAN_stacked.compile(loss=[_loss_generator, mean_squared_error,mean_squared_error,mean_squared_error,mean_squared_error], optimizer=optimizerD,loss_weights=[weight_of_vanilla_loss,1,1,1,1])
##############################################################################################################
print(' ')
print('Networks initialized.')
print(' ')

d_loss_list = g_loss_list = np.empty((0,6))

ROC_AUC_SCORE_list = np.empty((0,3))
ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[0, 1, 0]], axis=0)
best_ROC_AUC = 1
best_ROC_AUC_every_10 = np.empty(0)
best_ROC_AUC_every_10 = np.append(best_ROC_AUC_every_10, 1E30)


ROC_AUC_SCORE_list_p = np.empty((0,3))
ROC_AUC_SCORE_list_p = np.append(ROC_AUC_SCORE_list_p, [[0, 1, 0]], axis=0)
best_ROC_AUC_p = 1
best_ROC_AUC_every_10_p = np.empty(0)
best_ROC_AUC_every_10_p = np.append(best_ROC_AUC_every_10_p, 1E30)

list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))
print('Training files information:')
print('Location: %s%s'%(training_directory,training_name))
print('np.shape(list_of_training_files):', np.shape(list_of_training_files))
print(' ')

# list_of_training_files=['/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/Hard_map/hardmap_mu_data_0.npy']
file = np.random.choice(list_of_training_files, 1)

print('Loading initial training file:',file,'...')

X_train = np.load(file[0])

muon_weights, pdg_info, X_train, aux_values = np.split(X_train, [1,2,-4], axis=1)

if enhanced_GAN == True:
	aux_values = np.concatenate((np.ones((np.shape(aux_values)[0],1)),aux_values),axis=1)
	aux_values[:,0] = (aux_values[:,1]) + (aux_values[:,2]) + (aux_values[:,3]) + (aux_values[:,4])
	order = np.expand_dims(np.arange(0, np.shape(aux_values)[0]),1)
	aux_values = np.concatenate((aux_values,order),axis=1)
	aux_values_new = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(aux_values[:,0])))
	aux_values = aux_values[aux_values[:,0].argsort()]
	aux_values[:,0] = aux_values_new[aux_values_new.argsort()]
	aux_values = aux_values[aux_values[:,-1].argsort()]
	aux_weights, aux_values, order = np.split(aux_values,[1, -1],axis=1)
	muon_weights_train = np.squeeze(muon_weights)*np.squeeze(aux_weights)
	# if enhanced squared
	# muon_weights_train = np.squeeze(muon_weights)*np.squeeze(aux_weights)*np.squeeze(aux_weights)
	muon_weights_train = np.squeeze(muon_weights_train/np.sum(muon_weights_train))
	muon_weights_test = np.squeeze(muon_weights/np.sum(muon_weights))
else:
	muon_weights_train = np.squeeze(muon_weights)
	muon_weights_train = np.squeeze(muon_weights_train/np.sum(muon_weights_train))
	muon_weights_test = np.squeeze(muon_weights/np.sum(muon_weights))
list_for_np_choice = np.arange(np.shape(X_train)[0]) 

if enhanced_GAN == True:
	generator_aux_values = np.abs(np.random.normal(loc=0,scale=1,size=(np.shape(aux_values)[0],5)))
	generator_aux_values[:,0] = generator_aux_values[:,1] * generator_aux_values[:,2] * generator_aux_values[:,3] * generator_aux_values[:,4]
	generator_aux_values_new = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(generator_aux_values[:,0])))
	order = np.expand_dims(np.arange(0, np.shape(generator_aux_values)[0]),1)
	generator_aux_values = np.concatenate((generator_aux_values,order),axis=1)
	generator_aux_values = generator_aux_values[generator_aux_values[:,0].argsort()]
	generator_aux_values[:,0] = generator_aux_values_new[generator_aux_values_new.argsort()]
	generator_aux_values = generator_aux_values[generator_aux_values[:,-1].argsort()]
	generator_aux_weights, generator_aux_values, order = np.split(generator_aux_values,[1, -1],axis=1)
	# if enhanced squared:
	# generator_aux_weights = generator_aux_weights*generator_aux_weights
	generator_aux_weights = np.squeeze(generator_aux_weights/np.sum(generator_aux_weights))
	generator_pdg_info = np.random.choice([-1,1],size=np.shape(pdg_info),p=[1-Fraction_pos,Fraction_pos],replace=True)
else:
	generator_aux_values = np.abs(np.random.normal(loc=0,scale=1,size=(np.shape(aux_values)[0],4)))
	generator_aux_weights = np.ones(np.shape(generator_aux_values[:,0]))
	generator_aux_weights = np.squeeze(generator_aux_weights/np.sum(generator_aux_weights))
	generator_pdg_info = np.random.choice([-1,1],size=np.shape(pdg_info),p=[1-Fraction_pos,Fraction_pos],replace=True)


training_time = 0
every_10_save_index = 0
every_10_save_index_p = 0
t0 = time.time()

random_indicies = np.random.choice(list_for_np_choice, size=(3,100000), p=muon_weights_test, replace=False)

X_train_plot = X_train[random_indicies[0]]

axis_titles = ['StartX', 'StartY', 'StartZ', 'Px', 'Py', 'Pz']
plt.figure(figsize=(5*4, 3*4))
subplot=0
for i in range(0, 6):
	for j in range(i+1, 6):
		subplot += 1
		plt.subplot(3,5,subplot)
		plt.hist2d(X_train_plot[:,i], X_train_plot[:,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
		plt.xlabel(axis_titles[i])
		plt.ylabel(axis_titles[j])
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('%s%s/Correlations_TRAIN.png'%(working_directory,saving_directory))
plt.savefig('%s%s/correlations/Correlations_TRAIN.png'%(working_directory,saving_directory))
plt.close('all')

print(' ')
print('Begining training loop...')
print(' ')

for cnt in range(int(epochs)):
	if cnt == 1: print('Looping appears to be working correctly...')
	
	if cnt % 1000 == 0 and cnt > 0:
		batch += 1
	
	if cnt % 5000 == 0: 
		print(cnt)
		with open("%s%s/epoch.txt"%(working_directory,saving_directory), "a") as myfile:
			myfile.write('\n %d'%(cnt))

	if cnt % save_interval == 0 and cnt > 0: 
		list_of_training_files = glob.glob('%s%s'%(training_directory,training_name))
		# list_of_training_files=['/Volumes/Mac-Toshiba/PhD/Muon_Shield_Opt_GAN_sample/Hard_map/hardmap_mu_data_0.npy']
		file = np.random.choice(list_of_training_files, 1)
		print('Loading new training file:',file,'...')
		X_train = np.load(file[0])

		muon_weights, pdg_info, X_train, aux_values = np.split(X_train, [1,2,-4], axis=1)

		if enhanced_GAN == True:
			aux_values = np.concatenate((np.ones((np.shape(aux_values)[0],1)),aux_values),axis=1)
			aux_values[:,0] = (aux_values[:,1]) + (aux_values[:,2]) + (aux_values[:,3]) + (aux_values[:,4])
			order = np.expand_dims(np.arange(0, np.shape(aux_values)[0]),1)
			aux_values = np.concatenate((aux_values,order),axis=1)
			aux_values_new = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(aux_values[:,0])))
			aux_values = aux_values[aux_values[:,0].argsort()]
			aux_values[:,0] = aux_values_new[aux_values_new.argsort()]
			aux_values = aux_values[aux_values[:,-1].argsort()]
			aux_weights, aux_values, order = np.split(aux_values,[1, -1],axis=1)
			muon_weights_train = np.squeeze(muon_weights)*np.squeeze(aux_weights)
			# if enhanced squared
			# muon_weights_train = np.squeeze(muon_weights)*np.squeeze(aux_weights)*np.squeeze(aux_weights)
			muon_weights_train = np.squeeze(muon_weights_train/np.sum(muon_weights_train))
			muon_weights_test = np.squeeze(muon_weights/np.sum(muon_weights))
		else:
			muon_weights_train = np.squeeze(muon_weights)
			muon_weights_train = np.squeeze(muon_weights_train/np.sum(muon_weights_train))
			muon_weights_test = np.squeeze(muon_weights/np.sum(muon_weights))
		list_for_np_choice = np.arange(np.shape(X_train)[0]) 

		if enhanced_GAN == True:
			generator_aux_values = np.abs(np.random.normal(loc=0,scale=1,size=(np.shape(aux_values)[0],5)))
			generator_aux_values[:,0] = generator_aux_values[:,1] * generator_aux_values[:,2] * generator_aux_values[:,3] * generator_aux_values[:,4]
			generator_aux_values_new = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(generator_aux_values[:,0])))
			order = np.expand_dims(np.arange(0, np.shape(generator_aux_values)[0]),1)
			generator_aux_values = np.concatenate((generator_aux_values,order),axis=1)
			generator_aux_values = generator_aux_values[generator_aux_values[:,0].argsort()]
			generator_aux_values[:,0] = generator_aux_values_new[generator_aux_values_new.argsort()]
			generator_aux_values = generator_aux_values[generator_aux_values[:,-1].argsort()]
			generator_aux_weights, generator_aux_values, order = np.split(generator_aux_values,[1, -1],axis=1)
			# if enhanced squared:
			# generator_aux_weights = generator_aux_weights*generator_aux_weights
			generator_aux_weights = np.squeeze(generator_aux_weights/np.sum(generator_aux_weights))
			generator_pdg_info = np.random.choice([-1,1],size=np.shape(pdg_info),p=[1-Fraction_pos,Fraction_pos],replace=True)
		else:
			generator_aux_values = np.abs(np.random.normal(loc=0,scale=1,size=(np.shape(aux_values)[0],4)))
			generator_aux_weights = np.ones(np.shape(generator_aux_values[:,0]))
			generator_aux_weights = np.squeeze(generator_aux_weights/np.sum(generator_aux_weights))
			generator_pdg_info = np.random.choice([-1,1],size=np.shape(pdg_info),p=[1-Fraction_pos,Fraction_pos],replace=True)


	random_indicies = np.random.choice(list_for_np_choice, size=(3,int(batch)), p=muon_weights_train, replace=False)
	generator_random_indicies = np.random.choice(list_for_np_choice, size=(3,int(batch)), p=generator_aux_weights, replace=False)
	
	legit_images = np.expand_dims(np.concatenate((pdg_info[random_indicies[0]],X_train[random_indicies[0]]),axis=1),1)
	aux_legit = aux_values[random_indicies[0]]

	# generate aux_fake
	# 1-pick from training
	# 2-generate from cov with numbers for fraction
	if generator_noise_approach == 1:
		charge_fake = np.expand_dims(pdg_info[random_indicies[1]],1)
		aux_fake = aux_values[random_indicies[1]]
	elif generator_noise_approach == 2:
		charge_fake = np.expand_dims(generator_pdg_info[generator_random_indicies[1]],1)
		aux_fake = generator_aux_values[generator_random_indicies[1]]

	gen_noise = np.random.normal(0, 1, (int(batch), 100))
	syntetic_images = generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_fake,1), charge_fake])

	legit_labels = np.ones((int(batch), 1))
	gen_labels = np.zeros((int(batch), 1))
	
	aux_legit_i = np.expand_dims(aux_legit[:,0],1)
	aux_legit_j = np.expand_dims(aux_legit[:,1],1)
	aux_legit_k = np.expand_dims(aux_legit[:,2],1)
	aux_legit_l = np.expand_dims(aux_legit[:,3],1)

	aux_fake_i = np.expand_dims(aux_fake[:,0],1)
	aux_fake_j = np.expand_dims(aux_fake[:,1],1)
	aux_fake_k = np.expand_dims(aux_fake[:,2],1)
	aux_fake_l = np.expand_dims(aux_fake[:,3],1)

	d_loss_bin = discriminator_aux_xy.train_on_batch(legit_images, [aux_legit_i])
	d_loss_bin = discriminator_aux_z.train_on_batch(legit_images, [aux_legit_j])
	d_loss_bin = discriminator_aux_pxpy.train_on_batch(legit_images, [aux_legit_k])
	d_loss_bin = discriminator_aux_pz.train_on_batch(legit_images, [aux_legit_l])

	d_loss_legit = discriminator.train_on_batch(legit_images, [legit_labels, aux_legit_i, aux_legit_j, aux_legit_k, aux_legit_l])
	d_loss_gen = discriminator.train_on_batch(syntetic_images,[gen_labels, aux_fake_i, aux_fake_j, aux_fake_k, aux_fake_l])

	###################

	if generator_noise_approach == 1:
		charge_gan = np.expand_dims(pdg_info[random_indicies[2]],1)
		aux_gan = aux_values[random_indicies[2]]
	elif generator_noise_approach == 2:
		charge_gan = np.expand_dims(generator_pdg_info[generator_random_indicies[2]],1)
		aux_gan = generator_aux_values[generator_random_indicies[2]]

	gen_noise = np.random.normal(0, 1, (int(batch), 100))
	
	y_mislabled = np.ones((batch, 1))
	
	aux_gan_i = np.expand_dims(aux_gan[:,0],1)
	aux_gan_j = np.expand_dims(aux_gan[:,1],1)
	aux_gan_k = np.expand_dims(aux_gan[:,2],1)
	aux_gan_l = np.expand_dims(aux_gan[:,3],1)

	g_loss = GAN_stacked.train_on_batch([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan], [y_mislabled, aux_gan_i, aux_gan_j, aux_gan_k, aux_gan_l])

	d_loss_list = np.append(d_loss_list, [[cnt,(d_loss_legit[1]+d_loss_gen[1])/2, (d_loss_legit[2]+d_loss_gen[2])/2, (d_loss_legit[3]+d_loss_gen[3])/2, (d_loss_legit[4]+d_loss_gen[4])/2, (d_loss_legit[5]+d_loss_gen[5])/2]], axis=0)
	g_loss_list = np.append(g_loss_list, [[cnt, g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5]]], axis=0)
	

	if cnt % save_interval == 0:

		t1 = time.time()

		total = t1-t0

		training_time += total

		print('Saving ... ')

		if len(best_ROC_AUC_every_10) > 10 and cnt > save_interval: # Reset list every 10 saving steps
			best_ROC_AUC_every_10 = np.empty(0)
			best_ROC_AUC_every_10 = np.append(best_ROC_AUC_every_10, 1E30)
			every_10_save_index += 1

		if len(best_ROC_AUC_every_10_p) > 10 and cnt > save_interval: # Reset list every 10 saving steps
			best_ROC_AUC_every_10_p = np.empty(0)
			best_ROC_AUC_every_10_p = np.append(best_ROC_AUC_every_10_p, 1E30)
			every_10_save_index_p += 1


		plt.figure(figsize=(4*6,4*2))
		plt.subplot(2,6,1)
		if np.shape(d_loss_list)[0] > 1000:
			plt.title('D loss 1 = %.3f'%np.mean(d_loss_list[np.shape(d_loss_list)[0]-1000:,1]))
		plt.plot(d_loss_list[:,0],d_loss_list[:,1])
		plt.subplot(2,6,2)
		if np.shape(d_loss_list)[0] > 1000:
			plt.title('D loss 2 = %.3f'%np.mean(d_loss_list[np.shape(d_loss_list)[0]-1000:,2]))
		plt.plot(d_loss_list[:,0],d_loss_list[:,2])
		plt.subplot(2,6,3)
		if np.shape(d_loss_list)[0] > 1000:
			plt.title('D loss 3 = %.3f'%np.mean(d_loss_list[np.shape(d_loss_list)[0]-1000:,3]))
		plt.plot(d_loss_list[:,0],d_loss_list[:,3])
		plt.subplot(2,6,4)
		if np.shape(d_loss_list)[0] > 1000:
			plt.title('D loss 4 = %.3f'%np.mean(d_loss_list[np.shape(d_loss_list)[0]-1000:,4]))
		plt.plot(d_loss_list[:,0],d_loss_list[:,4])
		plt.subplot(2,6,5)
		if np.shape(d_loss_list)[0] > 1000:
			plt.title('D loss 5 = %.3f'%np.mean(d_loss_list[np.shape(d_loss_list)[0]-1000:,5]))
		plt.plot(d_loss_list[:,0],d_loss_list[:,5])
		# plt.subplot(2,6,6)
		# if np.shape(d_loss_list)[0] > 1000:
		# 	plt.title('D loss 6 = %.3f'%np.mean(d_loss_list[np.shape(d_loss_list)[0]-1000:,6]))
		# plt.plot(d_loss_list[:,0],d_loss_list[:,6])


		plt.subplot(2,6,7)
		if np.shape(g_loss_list)[0] > 1000:
			plt.title('G loss 1 = %.3f'%np.mean(g_loss_list[np.shape(g_loss_list)[0]-1000:,1]))
		plt.plot(g_loss_list[:,0],g_loss_list[:,1])
		plt.subplot(2,6,8)
		if np.shape(g_loss_list)[0] > 1000:
			plt.title('G loss 2 = %.3f'%np.mean(g_loss_list[np.shape(g_loss_list)[0]-1000:,2]))
		plt.plot(g_loss_list[:,0],g_loss_list[:,2])
		plt.subplot(2,6,9)
		if np.shape(g_loss_list)[0] > 1000:
			plt.title('G loss 3 = %.3f'%np.mean(g_loss_list[np.shape(g_loss_list)[0]-1000:,3]))
		plt.plot(g_loss_list[:,0],g_loss_list[:,3])
		plt.subplot(2,6,10)
		if np.shape(g_loss_list)[0] > 1000:
			plt.title('G loss 4 = %.3f'%np.mean(g_loss_list[np.shape(g_loss_list)[0]-1000:,4]))
		plt.plot(g_loss_list[:,0],g_loss_list[:,4])
		plt.subplot(2,6,11)
		if np.shape(g_loss_list)[0] > 1000:
			plt.title('G loss 5 = %.3f'%np.mean(g_loss_list[np.shape(g_loss_list)[0]-1000:,5]))
		plt.plot(g_loss_list[:,0],g_loss_list[:,5])
		# plt.subplot(2,6,12)
		# if np.shape(g_loss_list)[0] > 1000:
		# 	plt.title('G loss 6 = %.3f'%np.mean(g_loss_list[np.shape(g_loss_list)[0]-1000:,6]))
		# plt.plot(g_loss_list[:,0],g_loss_list[:,6])


		plt.savefig('%s%s/Loss.png'%(working_directory,saving_directory))
		plt.close('all')
	
		noise_size = 100000
		
		if cnt == 0: noise_size = 1000
			

		if generator_noise_approach == 1:
			random_indicies = np.random.choice(list_for_np_choice, size=(1,int(noise_size)), p=muon_weights_test, replace=False)
			charge_gan = np.expand_dims(pdg_info[random_indicies[0]],1)
			aux_gan = aux_values[random_indicies[0]]
		elif generator_noise_approach == 2:
			aux_gan = np.abs(np.random.normal(loc=0,scale=1,size=(noise_size,4)))
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(noise_size,1),p=[1-Fraction_pos,Fraction_pos],replace=True),1)
		gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
		images = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

		# Remove pdg info
		images = images[:,1:]

		########
		plt.figure(figsize=(5*4, 3*4))
		subplot=0
		for i in range(0, 6):
			for j in range(i+1, 6):
				subplot += 1
				plt.subplot(3,5,subplot)
				if subplot == 3: plt.title(cnt)
				plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
				plt.xlabel(axis_titles[i])
				plt.ylabel(axis_titles[j])
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s%s/Correlations.png'%(working_directory,saving_directory))
		plt.savefig('%s%s/correlations/Correlations_%d.png'%(working_directory,saving_directory,cnt))
		plt.close('all')

		#
		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 6):
			subplot += 1
			plt.subplot(2,3,subplot)
			if subplot == 2: plt.title(cnt)
			plt.hist([X_train_plot[:noise_size,i], images[:noise_size,i]], bins=50,range=[-1,1], label=['Train','GEN'],histtype='step')
			plt.xlabel(axis_titles[i])
			if axis_titles[i] == 'StartZ': plt.legend()
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s%s/Values.png'%(working_directory,saving_directory))
		plt.savefig('%s%s/correlations/Values%d.png'%(working_directory,saving_directory,cnt))
		plt.close('all')

		plt.figure(figsize=(3*4, 2*4))
		subplot=0
		for i in range(0, 6):
			subplot += 1
			plt.subplot(2,3,subplot)
			if subplot == 2: plt.title(cnt)
			plt.hist([X_train_plot[:noise_size,i], images[:noise_size,i]], bins=50,range=[-1,1], label=['Train','GEN'],histtype='step')
			plt.yscale('log')
			plt.xlabel(axis_titles[i])
			if axis_titles[i] == 'StartZ': plt.legend()
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s%s/Values_log.png'%(working_directory,saving_directory))
		plt.savefig('%s%s/correlations/Values%d_log.png'%(working_directory,saving_directory,cnt))
		plt.close('all')


		
		if generator_noise_approach == 1:
			random_indicies = np.random.choice(list_for_np_choice, size=(1,int(noise_size)), p=muon_weights_test, replace=False)
			charge_gan = np.expand_dims(pdg_info[random_indicies[0]],1)
			aux_gan = aux_values[random_indicies[0]]
			aux_gan[:,2] = aux_gan[:,2]*2.5
		elif generator_noise_approach == 2:
			aux_gan = np.abs(np.random.normal(loc=0,scale=1,size=(noise_size,4)))
			aux_gan[:,2] = aux_gan[:,2]*2.5
			charge_gan = np.expand_dims(np.random.choice([-1,1],size=(noise_size,1),p=[1-Fraction_pos,Fraction_pos],replace=True),1)
		gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
		images_wide = np.squeeze(generator.predict([np.expand_dims(gen_noise,1), np.expand_dims(aux_gan,1), charge_gan]))

		images_wide = images_wide[:,1:]
		#######
		plt.figure(figsize=(5*4, 3*4))
		subplot=0
		for i in range(0, 6):
			for j in range(i+1, 6):
				subplot += 1
				plt.subplot(3,5,subplot)
				if subplot == 3: plt.title(cnt)
				plt.hist2d(images_wide[:noise_size,i], images_wide[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
				plt.xlabel(axis_titles[i])
				plt.ylabel(axis_titles[j])
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig('%s%s/Correlations_2_5.png'%(working_directory,saving_directory))
		plt.savefig('%s%s/correlations/Correlations_2_5_%d.png'%(working_directory,saving_directory,cnt))
		plt.close('all')


		if cnt > 0:


			generator.save('%s%s/generator.h5'%(working_directory,saving_directory))
			discriminator.save('%s%s/discriminator.h5'%(working_directory,saving_directory))


			###################################################

			clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

			random_indicies = np.random.choice(list_for_np_choice, size=(noise_size), p=muon_weights_test, replace=False)
			X_train_sample = X_train[random_indicies,2:]

			bdt_train_size = int(np.shape(images)[0]/2)

			real_training_data = X_train_sample[:bdt_train_size]

			real_test_data = X_train_sample[bdt_train_size:]

			fake_training_data = np.squeeze(images[:bdt_train_size,2:])

			fake_test_data = np.squeeze(images[bdt_train_size:,2:])

			real_training_labels = np.ones(bdt_train_size)

			fake_training_labels = np.zeros(bdt_train_size)

			total_training_data = np.concatenate((real_training_data, fake_training_data))

			total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

			clf.fit(total_training_data, total_training_labels)

			out_real = clf.predict_proba(real_test_data)

			out_fake = clf.predict_proba(fake_test_data)

			plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
			plt.xlabel('Output of BDT')
			plt.legend(loc='upper right')
			plt.savefig('%s%s/bdt/BDT_P_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
			plt.close('all')

			ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

			ROC_AUC_SCORE_list_p = np.append(ROC_AUC_SCORE_list_p, [[cnt, ROC_AUC_SCORE_curr, training_time]], axis=0)

			if ROC_AUC_SCORE_list_p[-1][1] < best_ROC_AUC_p:
				print('Saving best ROC_AUC_P.')
				generator.save('%s%s/models/Generator_best_ROC_AUC_P.h5'%(working_directory,saving_directory))
				discriminator.save('%s%s/models/Discriminator_best_ROC_AUC_P.h5'%(working_directory,saving_directory))
				best_ROC_AUC_p = ROC_AUC_SCORE_list_p[-1][1]
				shutil.copy('%s%s/Correlations.png'%(working_directory,saving_directory), '%s%s/BEST_ROC_AUC_P_Correlations.png'%(working_directory,saving_directory))



			plt.figure(figsize=(8,4))
			plt.title('ROC_AUC_SCORE_list_p best: %.4f at %d'%(best_ROC_AUC_p,ROC_AUC_SCORE_list_p[np.where(ROC_AUC_SCORE_list_p==best_ROC_AUC_p)[0][0]][0]))
			plt.plot(ROC_AUC_SCORE_list_p[:,0],ROC_AUC_SCORE_list_p[:,1])
			plt.axhline(y=best_ROC_AUC_p,c='k',linestyle='--')
			plt.axvline(x=ROC_AUC_SCORE_list_p[np.where(ROC_AUC_SCORE_list_p==best_ROC_AUC_p)[0][0]][0],c='k',linestyle='--')
			plt.savefig('%s%s/ROC_progress_p.png'%(working_directory,saving_directory))
			plt.close('all')

			np.save('%s%s/models/FoM_ROC_AUC_SCORE_list_p'%(working_directory,saving_directory),ROC_AUC_SCORE_list_p)


			if ROC_AUC_SCORE_list_p[-1][1] < np.amin(best_ROC_AUC_every_10_p):
				print('Saving best ROC_AUC every 10.')
				with open("%s%s/models/best_ROC_AUC_every_10_p.txt"%(working_directory,saving_directory), "a") as myfile:
					myfile.write('\n %d, %.3f'%(cnt, ROC_AUC_SCORE_list_p[-1][1]))
				generator.save('%s%s/models/Generator_best_every_10_p_%d.h5'%(working_directory,saving_directory,every_10_save_index_p))
				discriminator.save('%s%s/models/Discriminator_best_every_10_p_%d.h5'%(working_directory,saving_directory,every_10_save_index_p))
			best_ROC_AUC_every_10_p = np.append(best_ROC_AUC_every_10_p, ROC_AUC_SCORE_list_p[-1][1])


			###################################################

			


			clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

			random_indicies = np.random.choice(list_for_np_choice, size=(noise_size), p=muon_weights_test, replace=False)
			X_train_sample = X_train[random_indicies]

			bdt_train_size = int(np.shape(images)[0]/2)

			X_train_sample[:,0] = np.add(X_train_sample[:,0],np.random.normal(loc=0,scale=0.00002,size=(np.shape(X_train_sample[:,0]))))
			X_train_sample[:,1] = np.add(X_train_sample[:,1],np.random.normal(loc=0,scale=0.00002,size=(np.shape(X_train_sample[:,1]))))

			images[:,0] = np.add(images[:,0],np.random.normal(loc=0,scale=0.00002,size=(np.shape(images[:,0]))))
			images[:,1] = np.add(images[:,1],np.random.normal(loc=0,scale=0.00002,size=(np.shape(images[:,1]))))

			real_training_data = X_train_sample[:bdt_train_size]

			real_test_data = X_train_sample[bdt_train_size:]

			fake_training_data = np.squeeze(images[:bdt_train_size])

			fake_test_data = np.squeeze(images[bdt_train_size:])

			real_training_labels = np.ones(bdt_train_size)

			fake_training_labels = np.zeros(bdt_train_size)

			total_training_data = np.concatenate((real_training_data, fake_training_data))

			total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

			clf.fit(total_training_data, total_training_labels)

			out_real = clf.predict_proba(real_test_data)

			out_fake = clf.predict_proba(fake_test_data)

			plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
			plt.xlabel('Output of BDT')
			plt.legend(loc='upper right')
			plt.savefig('%s%s/bdt/BDT_out_%d.png'%(working_directory,saving_directory,cnt), bbox_inches='tight')
			plt.close('all')

			ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

			ROC_AUC_SCORE_list = np.append(ROC_AUC_SCORE_list, [[cnt, ROC_AUC_SCORE_curr, training_time]], axis=0)

			if ROC_AUC_SCORE_list[-1][1] < best_ROC_AUC:
				print('Saving best ROC_AUC.')
				generator.save('%s%s/models/Generator_best_ROC_AUC.h5'%(working_directory,saving_directory))
				discriminator.save('%s%s/models/Discriminator_best_ROC_AUC.h5'%(working_directory,saving_directory))
				best_ROC_AUC = ROC_AUC_SCORE_list[-1][1]
				shutil.copy('%s%s/Correlations.png'%(working_directory,saving_directory), '%s%s/BEST_ROC_AUC_Correlations.png'%(working_directory,saving_directory))



			plt.figure(figsize=(8,4))
			plt.title('ROC_AUC_SCORE_list best: %.4f at %d'%(best_ROC_AUC,ROC_AUC_SCORE_list[np.where(ROC_AUC_SCORE_list==best_ROC_AUC)[0][0]][0]))
			plt.plot(ROC_AUC_SCORE_list[:,0],ROC_AUC_SCORE_list[:,1])
			plt.axhline(y=best_ROC_AUC,c='k',linestyle='--')
			plt.axvline(x=ROC_AUC_SCORE_list[np.where(ROC_AUC_SCORE_list==best_ROC_AUC)[0][0]][0],c='k',linestyle='--')
			plt.savefig('%s%s/ROC_progress.png'%(working_directory,saving_directory))
			plt.close('all')

			np.save('%s%s/models/FoM_ROC_AUC_SCORE_list'%(working_directory,saving_directory),ROC_AUC_SCORE_list)


			if ROC_AUC_SCORE_list[-1][1] < np.amin(best_ROC_AUC_every_10):
				print('Saving best ROC_AUC every 10.')
				with open("%s%s/models/best_ROC_AUC_every_10.txt"%(working_directory,saving_directory), "a") as myfile:
					myfile.write('\n %d, %.3f'%(cnt, ROC_AUC_SCORE_list[-1][1]))
				generator.save('%s%s/models/Generator_best_every_10_%d.h5'%(working_directory,saving_directory,every_10_save_index))
				discriminator.save('%s%s/models/Discriminator_best_every_10_%d.h5'%(working_directory,saving_directory,every_10_save_index))
			best_ROC_AUC_every_10 = np.append(best_ROC_AUC_every_10, ROC_AUC_SCORE_list[-1][1])


		print('Saving complete.')
		t0 = time.time()























