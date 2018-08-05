'''
Necessary library imports

'''

import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Input, Flatten
from keras.model import Models

def discriminator(type):

	input_layer = Input(shape=(256, 256, 3), name='input_layer')

	conv_1 = Conv2D(filters=64, kernel_size=(4, 4), strides=(type, type), activation='relu')(input_layer)
	
	conv_2 = Conv2D(filters=128, kernel_size=(4, 4), strides=(type, type))(conv_1)
	conv_2 = BatchNormalization(axis=-1)(conv_2)
	conv_2 = Activation('relu')(conv_2)
	
	conv_3 = Conv2D(filters=256, kernel_size=(4, 4), strides=(type, type))(conv_2)
	conv_3 = BatchNormalization(axis=-1)(conv_3)
	conv_3 =Activation('relu')(conv_3)

	conv_4 = Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 1))(conv_3)
	conv_4 = BatchNormalization(axis=-1)(conv_4)
	conv_4 = Activation('relu')(conv_4)

	conv_5 = Conv2D(filters=1, kernel_size=(4, 4),strides=(1, 1))(conv_4)

	output = Flatten()(conv_5)

	model = Model(inputs = input_layer, outputs = output)

	return model


