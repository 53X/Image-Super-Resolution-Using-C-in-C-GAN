import keras
from keras.layers import Conv2D, Input, merge
from keras.model import Model

def residual_block(input_tensor):

	'''
	This is code for the Residual block present in 
	generators of the proposed network. The Residual
	Block consists of 2 Conv2D layers and a merge layer
	for adding the output of the second Conv2D layer to
	the input tensor of the residual block.

	''' 

	conv_1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
					  padding='same', name='residual_1')(input_tensor)

	conv_1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
					  padding='same', name='residual_2')(conv_1_1)

	sum_tensor = merge([input_tensor, conv_1_2], mode='sum', name='residual_summation')

	return sum_tensor				  



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
					padding='same', name='first_conv')(input_layer)
	
	'''
	For G3, the kernel size and the stride is (4, 4) and (2, 2)
	respectively. For G1 and G2, the kernel size and the stride
	is (3, 3) and (1, 1)

	'''

	if(generator_type==3):

		k, s = 4, 2
	
	else:
		
		k, s = 3, 1

	
	conv_2 = Conv2D(filters=64, kernel_size=(k, k), strides=(s, s),
					padding='same', name='second_conv')(conv_1)
	conv_3 = Conv2D(filters=64, kernel_size=(k, k), strides=(s, s),
					padding='same', name='third_conv')(conv_2)

	input_tensor = conv_3
	
	'''
	Residual blocks added for 6 consecutive times

	''' 

	for i in range(6):

		residual_output = residual_block(input_tensor)
		input_tensor = residual_output

	conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
					padding='same', name='fourth_conv')(residual_output)	

	conv_5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
					padding='same', name='fifth_conv')(conv_4)

	output_generator = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1),
							  padding='same', name='generator_output')(conv_5)

	model = Model(inputs=input_layer, outputs=output_generator)
	
	return model						  










					