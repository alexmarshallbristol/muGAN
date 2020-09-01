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
		
def create_discriminator_SMEAR():
	''' Due to the use of Lambda layers the disciminator module cannot be loaded normally as it was trained on a different Keras version.
	Instead we recreate it here then simply load the weights. '''

	D_architecture = [1500,2000]
	D_architecture_xy = D_architecture_z = D_architecture_pxpy = D_architecture_pz = D_architecture_x = D_architecture_y = D_architecture_px = D_architecture_py = [32, 64]

	##############################################################################################################
	# Build Discriminator xy model ...

	# print(' ')
	# print('Discriminator...')
	# print(' ')

	##############################################################################################################
	# Build Discriminator x model ...

	d_input = Input(shape=(1,7))

	H_x = split_tensor(1, d_input)
	H_x = Reshape((1,1))(H_x)

	H = Flatten()(H_x)

	for layer in D_architecture_x:

		H = Dense(int(layer))(H)
		H = LeakyReLU(alpha=0.2)(H)
		H = Dropout(0.2)(H)

	d_output_aux = Dense(1, activation='relu')(H)

	discriminator_aux_x = Model(d_input, [d_output_aux])

	# discriminator_aux_x.compile(loss=[mean_squared_error],optimizer=optimizerD)
	##############################################################################################################

	##############################################################################################################
	# Build Discriminator x model ...

	d_input = Input(shape=(1,7))

	H_y = split_tensor(2, d_input)
	H_y = Reshape((1,1))(H_y)

	H = Flatten()(H_y)

	for layer in D_architecture_y:

		H = Dense(int(layer))(H)
		H = LeakyReLU(alpha=0.2)(H)
		H = Dropout(0.2)(H)

	d_output_aux = Dense(1, activation='relu')(H)

	discriminator_aux_y = Model(d_input, [d_output_aux])

	# discriminator_aux_y.compile(loss=[mean_squared_error],optimizer=optimizerD)
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

	# discriminator_aux_z.compile(loss=[mean_squared_error],optimizer=optimizerD)
	##############################################################################################################


	##############################################################################################################
	# Build Discriminator px model ...

	d_input = Input(shape=(1,7))

	H_px = split_tensor(4, d_input)
	H_px = Reshape((1,1))(H_px)

	H = Flatten()(H_px)

	for layer in D_architecture_px:

		H = Dense(int(layer))(H)
		H = LeakyReLU(alpha=0.2)(H)
		H = Dropout(0.2)(H)

	d_output_aux = Dense(1, activation='relu')(H)

	discriminator_aux_px = Model(d_input, [d_output_aux])

	# discriminator_aux_px.compile(loss=[mean_squared_error],optimizer=optimizerD)
	##############################################################################################################


	##############################################################################################################
	# Build Discriminator py model ...

	d_input = Input(shape=(1,7))

	H_py = split_tensor(5, d_input)
	H_py = Reshape((1,1))(H_py)

	H = Flatten()(H_py)

	for layer in D_architecture_py:

		H = Dense(int(layer))(H)
		H = LeakyReLU(alpha=0.2)(H)
		H = Dropout(0.2)(H)

	d_output_aux = Dense(1, activation='relu')(H)

	discriminator_aux_py = Model(d_input, [d_output_aux])

	# discriminator_aux_py.compile(loss=[mean_squared_error],optimizer=optimizerD)
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

	# discriminator_aux_pz.compile(loss=[mean_squared_error],optimizer=optimizerD)
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

	make_trainable(discriminator_aux_x, False)
	make_trainable(discriminator_aux_y, False)
	make_trainable(discriminator_aux_z, False)
	make_trainable(discriminator_aux_px, False)
	make_trainable(discriminator_aux_py, False)
	make_trainable(discriminator_aux_pz, False)

	d_output_aux_i = discriminator_aux_x(d_input)
	d_output_aux_j = discriminator_aux_y(d_input)
	d_output_aux_k = discriminator_aux_z(d_input)
	d_output_aux_l = discriminator_aux_px(d_input)
	d_output_aux_m = discriminator_aux_py(d_input)
	d_output_aux_n = discriminator_aux_pz(d_input)

	discriminator = Model(d_input, [d_output, d_output_aux_i, d_output_aux_j, d_output_aux_k, d_output_aux_l, d_output_aux_m, d_output_aux_n])


	return discriminator