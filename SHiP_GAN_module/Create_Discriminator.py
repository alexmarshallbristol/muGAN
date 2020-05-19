from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, BatchNormalization, Concatenate, Lambda, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 


def split_tensor(index, x):
    return Lambda(lambda x : x[:,:,index])(x)

def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val
		
def create_discriminator():
	''' Due to the use of Lambda layers the disciminator module cannot be loaded normally as it was trained on a different Keras version.
	Instead we recreate it here then simply load the weights. '''

	D_architecture = [1500,2000]
	D_architecture_xy = D_architecture_z = D_architecture_pxpy = D_architecture_pz = [32, 64]

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

	return discriminator