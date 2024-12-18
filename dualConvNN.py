from ConvNN import ConvNeuralNetwork
from tensorflow.keras import layers, Input, Model, activations

class dualConvNeuralNetwork(ConvNeuralNetwork):
	def __init__(self):
		super().__init__()
	
	def __init__(self, data, nNodes, name = "model",channels = ["E","t","r"]):
		super().__init__(data, nNodes, name, channels)

	#dual conv NNs with different masks - both are tuples
	def BuildModel(self, mask1, mask2):
		input_layer = Input(shape=self._xtrain[0].shape)
	
		#first CNN
		kernel_dim_1 = mask1
		#reLu activation at internal layers
		conv_layers_1 = [layers.Conv2D(n,kernel_dim_1,name="first_conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x_1 = conv_layers_1[0](input_layer)
		for i, c in enumerate(conv_layers_1[1:]):
			x_1 = c(x_1)

		#second CNN
		kernel_dim_2 = mask2
		#reLu activation at internal layers
		conv_layers_2 = [layers.Conv2D(n,kernel_dim_2,name="second_conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x_2 = conv_layers_2[0](input_layer)
		for i, c in enumerate(conv_layers_2[1:]):
			x_2 = c(x_2)
	
		#put into (None, 1, 1, nNodes) shape
		kernel_dim_2 = (x_2.shape[1], x_2.shape[2])
		x_2 = layers.Conv2D(self._nNodes[-1],kernel_dim_2,name="second_conv_layer_last",activation=activations.relu)(x_2)

		x = layers.concatenate([x_spikes, x_2])	

		#flatten from 2D to 1D
		x = layers.Flatten()(x)
		x = layers.Dense(64,name="dense_layer_1",activation=activations.relu)(x)
		x = layers.Dense(64,name="dense_layer_2",activation=activations.relu)(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(len(self._ytrain[0]),activation=activations.softmax,name="output")
		
		x = output_layer(x) 
		self._model = Model(inputs = input_layer, outputs = x, name = self._name)
