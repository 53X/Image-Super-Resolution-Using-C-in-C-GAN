import keras
from keras.layers import Conv2D, Input, merge

def residual_block(input_tensor):

	'''
	This is code for the Residual block present in 
	generators of the proposed network. The Residual
	Block consists of 2 Conv2D layers and a merge layer
	for adding the output of the second Conv2D layer to
	the input tensor of the residual block.

	''' 

	conv_1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
					  padding='valid', name='residual_1')(input_tensor)

	conv_1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
					  padding='valid', name='residual_2')(conv_1_1)
					  				  















def generator(generator_type):

	''' 
	This is the code for the Generators G1, G2, G3 as per the paper.
	Although the generators share the same architecture ,there is a
	little variation in the architectures of the [G1, G2] and [G3].
	However all the are made of 3 Conv2D layers on the either side of
	the network and contain 6 residual blocks in between.

	'''

	input_layer = Input(shape(32,32,1),name='LowResInput')
	conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1),
					padding='valid', name='first_conv')(input_layer)
	
	if(generator_type==3):

		k, s = 4, 2
	
	else:
		
		k, s = 3, 1

	
	conv_2 = Conv2D(filters=64, kernel_size=(k, k), strides=(s, s),
					padding='valid', name='second_conv')(conv_1)
	conv_3 = Conv2D(filters=64, kernel_size=(k, k), strides=(s, s),
					padding='valid', name='first_conv')(conv_2)



					