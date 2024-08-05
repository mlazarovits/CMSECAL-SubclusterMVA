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

	def summary(self):
		self._model.summary()

	#fully connected network
	def BuildModel(self):
		input_layer = Input(shape=(1,self._inputShape))
		
		#reLu activation at internal layers
		dense_layers = [layers.Dense(n,name="dense_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x = dense_layers[0](input_layer)
		for i, d in enumerate(dense_layers[1:]):
			x = d(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(1,activation=activations.sigmoid)
		
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


	def VizLoss(self, history, fname):
		plt.figure()
		plt.plot(history.history['val_loss'], label="val loss")
		plt.plot(history.history['loss'],label="train loss")
		plt.title("Loss")
		plt.legend()
		plt.savefig("loss/"+fname)

	def TrainModel(self,x,y,viz:bool = False,verb= 0):
		his = self._model.fit(x,y,epochs=1,verbose=verb)
		if viz:
			self.VizLoss(his,"test")
