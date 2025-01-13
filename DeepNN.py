from ModelBase import ModelBase
from keras import layers, metrics, Input, Model, activations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler, LabelBinarizer
import os
import subprocess
import numpy as np

class DeepNeuralNetwork(ModelBase):
	def __init__(self):
		self._inputShape = None
		self._nNodes = None
		self._xtrain = None
		self._ytrain = None
		#plot format
		self._form = "pdf"
		self._path = "results/"
		self._features = []
		self._bestModel = None
		self._lowestValLoss = 999
		self._catnames = {}
		self._catcolors = {}
		self._lb = None
		self._scaler = None
		self._inputHists = None
		super().__init__()

	def __init__(self, data, nNodes, name = "model"):
		self._bestModel = None
		self._lowestValLoss = 999
		self._form = "pdf"
		self._name = name
		self._path = "results/"+self._name
		self._catnames = {} 
		self._catcolors = {}
		self._inputHists = []
		if not os.path.exists(self._path):
			os.mkdir(self._path)
		#a list of ints that defines the nodes for each dense layer (obviously len(nNodes) == # layers
		self._nNodes = nNodes
		
		self._lb = LabelBinarizer()
		labels = data["label"]
		y = self._lb.fit_transform(labels)
		
		#extract inputs and labels, remove unnecessary columns
		#drop event + subcl cols
		dropcols = ["sample","event","object","subcl","label"]
		x = data.drop(dropcols,axis=1)

		self._features = x.columns
		x = x.to_numpy()
		#print("unnorm",x[0:5],max(x[:,0]))
		##normalize data
		self._scaler = MinMaxScaler()
		self._scaler.fit(x)
		x = self._scaler.transform(x) 
		#print("norm",x[:5],max(x[:,0]))
		#80/20 train/test split
		rand = 43 #change to random number to randomize
		self._xtrain, self._xtest, self._ytrain, self._ytest = train_test_split(x,y,test_size=0.2,random_state=rand)
		self._ytrain = np.asarray([ np.asarray(i) for i in self._ytrain])
		#print(self._xtrain.shape[0],"training samples",self._ytrain.shape,type(self._ytrain),type(self._ytrain[0]),self._ytrain[0])
		#shape of input data
		super().__init__()

	#fully connected network
	def BuildModel(self):
		input_layer = Input(shape=(self._xtrain.shape[1],))
		#reLu activation at internal layers
		dense_layers = [layers.Dense(n,name="dense_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x = dense_layers[0](input_layer)
		for i, d in enumerate(dense_layers[1:]):
			x = d(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(len(self._ytrain[0]),activation=activations.softmax,name="output")
		
		x = output_layer(x) 
		self._model = Model(inputs = input_layer, outputs = x, name = self._name)
		#print("inputs")
		#[print(i.shape, i.dtype) for i in self._model.inputs]
		#print("outputs")
		#[print(o.shape, o.dtype) for o in self._model.outputs]
		#print("layers")
		#[print(l.name, l.input_shape, l.dtype) for l in self._model.layers]
		#print()

	#SGD optimizer
	#cross-entropy loss (binary or categorical depending on labeling scheme)
	#monitor metrics: accuracy
	def CompileModel(self):
		self._model.compile(
			optimizer = 'adam',
			loss = 'categorical_crossentropy',
			metrics = ['accuracy','AUC']
		)


	def VizInputs(self):
		labels = self._lb.classes_
		all_labels = self._lb.inverse_transform(self._ytrain)
		inputs = [[[] for l in labels] for f in self._features]
		xtrain = self._scaler.inverse_transform(self._xtrain)
		for i, f in enumerate(inputs):
			self._inputHists.append([])
			plotname = self._path+"/"+self._features[i]+"."+self._form
			for j, x in enumerate(xtrain):
				#print(i,x,self._ytrain[j],self._features[i])
				#this sample needs to be put in j == label[k]
				#print(self._xtrain[j],self._ytrain[j],all_labels[j])
				lidx = np.flatnonzero(self._lb.classes_ == all_labels[j])[0]
				#print(i,self._features[i])
				inputs[i][lidx].append(x[i])
			for j, l in enumerate(labels):
				ns, bins, _ = plt.hist(inputs[i][j],label=self._catnames[labels[j]],log=True,bins=50,histtype=u'step')
				#self._inputHists[feature][label][ns, bins][bin #]
				bindict = {}
				bindict["ns"] = ns
				bindict["bins"] = bins
				self._inputHists[i].append(bindict)
			if os.path.exists(self._path+"/"+self._features[i]+"."+self._form):
				continue
			plt.title(self._features[i])
			plt.legend()
			print("Saving "+self._features[i]+" plot to",plotname)
			plt.savefig(plotname,format=self._form)
			plt.close()		

	#closure test - remake distributions of input features with weights applied to training samples
	def ValidateModel(self):
		samp_weights = np.array(self._model.predict(self._xtrain))
		#possible labels
		labels = self._lb.classes_
		all_labels = self._lb.inverse_transform(self._ytrain)
		inputs = [[[] for l in labels] for f in self._features]
		xtrain = np.array(self._scaler.inverse_transform(self._xtrain))
		for i, f in enumerate(inputs):
			plotname = self._path+"/"+self._features[i]+"_pred."+self._form
			fig, (ax1, ax2) = plt.subplots(nrows=2)
			ax1.grid(True)
			ax2.grid(True)
			legend_elements_ax1 = [
					Patch(fill=False,edgecolor='black',
                        			label='pred'),
					Patch(facecolor='black',edgecolor='black', alpha=0.5,
                        			label='true')]
			legend_elements_ax2 = []
			for j, l in enumerate(labels):
				ax1.stairs(self._inputHists[i][j]["ns"],self._inputHists[i][j]["bins"],color=self._catcolors[labels[j]],fill=True,alpha=0.5)
				ns, bins, _ = ax1.hist(xtrain[:,i],label=self._catnames[labels[j]],weights=samp_weights[:,j],bins=self._inputHists[i][j]["bins"],histtype=u'step',color=self._catcolors[labels[j]],fill=False)
				#replace zero values to negative number -> sets ratio to -1 -> avoids divide by zero
				ymax = -999
				for idx, n in enumerate(self._inputHists[i][j]["ns"]):
					if n == 0 or ns[idx] == 0:
						ns[idx] = 1
						self._inputHists[i][j]["ns"][idx] = -ns[idx]
					if n/ns[idx] > ymax:
						ymax = n/ns[idx]
					if np.isnan(n/ns[idx]):
						print("idx",idx,"div",ns[idx] / n,"ns",ns[idx],"input",n)
				ax2.scatter(bins[:-1],ns / self._inputHists[i][j]["ns"],color=self._catcolors[labels[j]],s=[8 for n in ns])
				ax2.set_ylabel("pred/true")
				legend_elements_ax2.append(Patch(fill=False,edgecolor=self._catcolors[labels[j]],
                        			label=self._catnames[labels[j]]))
			ax2.set_ylim([0,ymax]) #wont plot negative numbers -> no entry in denom
			fig.suptitle("predicted and true "+self._features[i])
			ax1.set_yscale('log')
			ax1.legend(handles = legend_elements_ax1)
			ax2.legend(handles = legend_elements_ax2)
			print("Saving predicted "+self._features[i]+" plot to",plotname)
			fig.savefig(plotname,format=self._form)
