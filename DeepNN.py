from ModelBase import ModelBase
from tensorflow.python.keras import layers, metrics, Input, Model, activations

class DeepNeuralNetwork(ModelBase):
	def __init__(self):
		self._inputShape = None
		self._nNodes = None
		super().__init__()

	def __init__(self, inputShape, nNodes):
		#shape of input data
		self._inputShape = inputShape
		#a list of ints that defines the nodes for each dense layer (obviously len(nNodes) == # layers
		self._nNodes = nNodes
		super().__init__()

	#fully connected network
	def BuildModel(self):
		input_layer = Input(shape=(1,self._inputShape))
		#reLu activation at internal layers
		dense_layers = [layers.Dense(n,name="dense_layer"+str(n),activation=activatons.relu) for n in nNodes]
		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(1,activation=activations.sigmoid)

		x = dense_layers[0](input_layer)
		for d in dense_layers[:1]:
			x = d(x)

		x = output_layer(x) 
		self._model = Model(inputs = input_layer, outputs = x)

	#SGD optimizer
	#cross-entropy loss (binary or categorical depending on labeling scheme)
	#monitor metrics: accuracy
	def CompileModel(self):
		self._model.compile(
			optimizer = 'sgd',
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
		)



