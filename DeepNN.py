from ModelBase import ModelBase
from keras import layers, metrics, Input, Model, activations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
import os

class DeepNeuralNetwork(ModelBase):
	def __init__(self):
		self._inputShape = None
		self._nNodes = None
		self._x = None
		self._y = None
		super().__init__()

	def __init__(self, data, nNodes):
		#a list of ints that defines the nodes for each dense layer (obviously len(nNodes) == # layers
		self._nNodes = nNodes
		#extract inputs and labels, remove unnecessary columns
		#drop event + subcl cols
		dropcols = ["Event","subcl","label"]
		#TODO:binary labels for now, will have to one-hot encode for multiclass
		y = data["label"].to_numpy()
		x = data.drop(dropcols,axis=1).to_numpy()
		#80/20 train/test split
		rand = 43 #change to random number to randomize
		self._xtrain, self._xtest, self._ytrain, self._ytrue = train_test_split(x,y,test_size=0.2,random_state=rand)
		print(self._xtrain.shape[0],"training samples")
		#shape of input data
		super().__init__()

	def summary(self):
		self._model.summary()

	#fully connected network
	def BuildModel(self,name = "model"):
		input_layer = Input(shape=(self._xtrain.shape[1],))
		
		#reLu activation at internal layers
		dense_layers = [layers.Dense(n,name="dense_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x = dense_layers[0](input_layer)
		for i, d in enumerate(dense_layers[1:]):
			x = d(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(1,activation=activations.sigmoid)
		
		x = output_layer(x) 
		self._model = Model(inputs = input_layer, outputs = x, name = name)

	#SGD optimizer
	#cross-entropy loss (binary or categorical depending on labeling scheme)
	#monitor metrics: accuracy
	def CompileModel(self):
		self._model.compile(
			optimizer = 'sgd',
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
		)


	def VizMetric(self, history, fname):
		plt.figure()
		plt.plot(history.history['val_'+fname], label="val "+fname)
		plt.plot(history.history[fname],label="train "+fname)
		plt.title(fname)
		plt.legend()
		#check if output dir exists
		path = "results/"+self._model.name
		if not os.path.exists(path):
			os.mkdir(path)
		plt.savefig(path+"/"+fname)


	#Caltech delayed photon analysis just plots fpr vs tpr for their DNN performance
	#TODO:labels are on binary for now (0, 1) but will eventually be (0, 1, 2) for (sig, spike, BH)
	#see https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#roc-curve-showing-a-specific-class
	#for multiclass ROC (one-vs-rest = sig-vs-rest)
	def VizROC(self, ytrue, ypred):
		display = RocCurveDisplay.from_predictions(ytrue,ypred,
			name="signal vs rest",
			color="pink",
			plot_chance_level=True,
		)
		display.ax_.set(
			xlabel="False Positive Rate",
			ylabel="True Positive Rate",
			title="Signal vs (spike & beam halo) subcluster ROC"
		)
		#check if output dir exists
		path = "results/"+self._model.name
		if not os.path.exists(path):
			os.mkdir(path)
		plt.savefig(path+"/ROCplot")

	def TrainModel(self,epochs=1,viz=False,verb=1):
		#80/20 train/val split (of training data)
		his = self._model.fit(self._xtrain,self._ytrain,epochs=epochs,verbose=verb,validation_split=0.2)
		if viz:
			self.VizMetric(his,"loss")
			#can also add accuracy, etc.

	def TestModel(self,batch_size=1,viz=False,verb=1):
		ypred = self._model.predict(self._xtest,batch_size=batch_size,verbose=verb)
		print("ytrue",self._ytrue,"ypred",ypred)
		if viz:
			self.VizROC(self._ytrue, ypred[0])
				
