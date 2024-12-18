from ConvNN import ConvNeuralNetwork
from tensorflow.keras import layers, Input, Model, activations

class dualConvNeuralNetwork(ConvNeuralNetwork):
	def __init__(self):
		super().__init__()
	
	def __init__(self, data, nNodes, name = "model",channels = ["E","t","r"]):
		super().__init__(data, nNodes, name, channels)

	#dual conv NNs with different masks - one for spikes, one for beam halo
	def BuildModel(self):
		input_layer = Input(shape=self._xtrain[0].shape)
	
		#n filters with 3x3 kernels - spikes
		kernel_dim_spikes = 3
		#reLu activation at internal layers
		conv_layers_spikes = [layers.Conv2D(n,kernel_dim_spikes,name="spike_conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x_spikes = conv_layers_spikes[0](input_layer)
		for i, c in enumerate(conv_layers_spikes[1:]):
			x_spikes = c(x_spikes)

		#n filters with 5x3 kernels - spikes
		kernel_dim_bh = (3,1)
		#reLu activation at internal layers
		conv_layers_bh = [layers.Conv2D(n,kernel_dim_bh,name="bh_conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x_bh = conv_layers_bh[0](input_layer)
		for i, c in enumerate(conv_layers_bh[1:]):
			x_bh = c(x_bh)
	
		#put into (None, 1, 1, nNodes) shape
		kernel_dim_bh = (x_bh.shape[1], x_bh.shape[2])
		x_bh = layers.Conv2D(self._nNodes[-1],kernel_dim_bh,name="bh_conv_layer_last",activation=activations.relu)(x_bh)

		x = layers.concatenate([x_spikes, x_bh])	

		#flatten from 2D to 1D
		x = layers.Flatten()(x)
		x = layers.Dense(64,name="dense_layer_1",activation=activations.relu)(x)
		x = layers.Dense(64,name="dense_layer_2",activation=activations.relu)(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(len(self._ytrain[0]),activation=activations.softmax,name="output")
		
		x = output_layer(x) 
		self._model = Model(inputs = input_layer, outputs = x, name = self._name)
