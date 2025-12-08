from ModelBase import ModelBase
from keras import layers, metrics, Input, Model, activations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler, OneHotEncoder
import os
import subprocess
import numpy as np
import pandas as pd

class DeepNeuralNetwork(ModelBase):
	def __init__(self):
		self._inputShape = None
		self._nNodes = None
		self._xtrain = None
		self._ytrain = None
		self._wtrain = None
		self._xtest = None
		self._ytest = None
		self._wtest = None
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

	def __init__(self, data, nNodes, cols, catnames, catcolors, name = "model"):
		self._bestModel = None
		self._lowestValLoss = 999
		self._form = "pdf"
		self._name = name
		self._path = "results/"+self._name
		self._catnames = catnames 
		self._catcolors = catcolors
		self._inputHists = []
		if not os.path.exists(self._path):
			os.mkdir(self._path)
		#a list of ints that defines the nodes for each dense layer (obviously len(nNodes) == # layers
		self._nNodes = nNodes
	
		self._dropcols = ["sample","event","object","label","Energy"]
		self._xtrain = None
		self._ytrain = None
		self._wtrain = None
		self._xtest = None
		self._ytest = None
		self._wtest = None
		self._xtrain_energy = None
		self._xtest_energy = None
		self._ytrain_energy = None
		self._ytest_energy = None 
		rand = 43 #change to random number to randomize
		#fir onehot enocoder
		self._lb = OneHotEncoder(sparse_output=False)
		labels = data["label"].to_numpy()
		labels = labels.reshape(-1,1)
		self._lb.fit(labels)
		self._features = [i for i in cols if i not in self._dropcols] 
		y = self.StripLabels(data)
		x, w, energy = self.ProcessData(data)
		#normalize data
		self._scaler = MinMaxScaler()
		self._scaler.fit(x)
		x = self._scaler.transform(x)
		#print("norm",x[:5],max(x[:,0]))
		#80/20 train/test split
		#if weights have been specified
		if(len(w) > 0):
			self._xtrain, self._xtest, self._ytrain, self._ytest, self._wtrain, self._wtest = train_test_split(data,y,w,test_size=0.2,random_state=rand)
		else:
			self._xtrain, self._xtest, self._ytrain, self._ytest = train_test_split(data,y,test_size=0.2,random_state=rand)
		if(len(energy) > 0):
			self._xtrain_energy, self._xtest_energy, self._ytrain_energy, self._ytest_energy = train_test_split(energy, y, test_size = 0.2, random_state = rand)
		self._ytrain = np.asarray([ np.asarray(i) for i in self._ytrain])
		
		#make hists of training data
		indata = pd.DataFrame(data=self._scaler.inverse_transform(self._xtrain),columns=self._features)
		indata['label'] = self._lb.inverse_transform(self._ytrain)
		self.MakeHists(indata,self._features,catnames) 
	
		#super().__init__()

	def StripLabels(self, indata):
		labels = indata["label"].to_numpy()
		labels = labels.reshape(-1,1)
		#print("labels",labels)
		y = self._lb.transform(labels)
		indata.drop("label",axis=1,inplace=True)
		return y
		

	def MakeSamples(self, indata):
		#remove dropcols from cols to pass to hist maker
		dropcols = [] 
		#print("features",self._features)	
		#print("data",data.shape)	

		#print("labels",np.unique(labels),"transformed labels",y,"catnames",self._catnames,"classes",self._lb.categories_)
	
		#extract inputs and labels, remove unnecessary columns
		#drop event + subcl cols
		energy = np.array([])
		if "Energy" in indata.columns:
			dropcols.append("Energy")
			energy = indata["Energy"].to_numpy()
			indata = indata.drop("Energy",axis=1)
			
		weights = np.array([])
		if("weight" in indata.columns):
			weights = indata["weight"].to_numpy()
			dropcols.append("weight")
			indata = indata.drop("weight",axis=1)

		indata = indata[self._features] 
		x = indata.to_numpy()
		#print("x",x.shape,x[0],"y",y[0],"indata[labels]",labels[0])	
		return x, weights, energy



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

	#SGD optimizer
	#cross-entropy loss (binary or categorical depending on labeling scheme)
	#monitor metrics: accuracy
	def CompileModel(self):
		self._model.compile(
			optimizer = 'adam',
			loss = 'categorical_crossentropy',
			metrics = ['accuracy','AUC']
		)


	def VizSamples(self):
		labels = self._lb.categories_[0]
		all_labels = self._lb.inverse_transform(self._ytrain)
		inputs = [[[] for l in labels] for f in self._features]
		weights = [[] for l in labels]
		xtrain = self._scaler.inverse_transform(self._xtrain)
		for i, f in enumerate(inputs):
			self._inputHists.append([])
			plotname = self._path+"/"+self._features[i]+"."+self._form
			bins = np.linspace(xtrain[:,i].min(), xtrain[:,i].max(), 50)
			for j, x in enumerate(xtrain):
				#this sample needs to be put in j == label[k]
				lidx = np.flatnonzero(labels == all_labels[j])[0]
				inputs[i][lidx].append(x[i])
				if(i == 0 and self._wtrain is not None):
					weights[lidx].append(self._wtrain[j])
			for j, l in enumerate(labels):
				if(self._wtrain is not None):
					ns, bins, _ = plt.hist(inputs[i][j],label=self._catnames[l],log=True,bins=bins,histtype=u'step',weights=weights[j])
				else:
					ns, bins, _ = plt.hist(inputs[i][j],label=self._catnames[l],log=True,bins=bins,histtype=u'step')
				#self._inputHists[feature][label][ns, bins][bin #]
				bindict = {}
				bindict["ns"] = ns
				bindict["bins"] = bins
				self._inputHists[i].append(bindict)
			if os.path.exists(self._path+"/"+self._features[i]+"."+self._form): #need to still create the hists for _inputHists closure test
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
		labels = self._lb.categories_[0]
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






