from ModelBase import ModelBase
from keras import layers, metrics, Input, Model, activations 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
import os
import subprocess
import numpy as np
import pandas as pd

class DeepNeuralNetwork(ModelBase):
	def __init__(self):
		super().__init__()
		self._inputShape = None
		self._nNodes = None
		self._xtrain = None
		self._xtrain_mu = None
		self._xtrain_var = None
		self._ytrain = None
		self._wtrain = None
		self._xtest = None
		self._ytest = None
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
		self._obj = "Photon"

	def __init__(self, data, nNodes, cols, catnames, catcolors, name = "model", extra = ""):
		super().__init__()
		self._bestModel = None
		self._lowestValLoss = 999
		self._form = "pdf"
		self._name = name
		self._path = "results/"+self._name
		self._catnames = catnames 
		self._obj = "Photon"
		self._catcolors = catcolors
		self._inputHists = []
		self._extra_label = extra
		if not os.path.exists(self._path):
			os.mkdir(self._path)
		self._discr_info = self._path+"/discr_info"
		if self._extra_label != "":
			self._discr_info += "_"+self._extra_label
		self._discr_info += ".txt"
		with open(self._discr_info, "w") as f:
			f.write("Network: "+self._name)
		#a list of ints that defines the nodes for each dense layer (obviously len(nNodes) == # layers
		self._nNodes = nNodes
	
		self._xtrain = None
		self._ytrain = None
		self._xtest = None
		self._ytest = None
		rand = 43 #change to random number to randomize
		#fir onehot enocoder
		self._features = cols 
		print("self._features",self._features)
		if data is not None:
			x, y = self.StripLabels(data)
			#80/20 train/test split
			#samples are dataframes
			self._xtrain_df, self._xtest_df, self._ytrain, self._ytest = train_test_split(x,y,test_size=0.2,random_state=rand)
			self._xtrain_df.reset_index()
			self._xtest_df.reset_index()

			#test_sample = self._xtest_df["sample"].to_numpy()
			#flatten
			self._ytrain = self._ytrain.flatten() 
			labels = self._lb.inverse_transform(self._ytrain)
			self._xtrain_df['label'] = labels
			labels = self._lb.inverse_transform(self._ytest)
			self._xtest_df['label'] = labels
	
			self._xtrain = self.MakeSamples(self._xtrain_df)
			self._xtrain_mu = self._xtrain.mean(axis=0)
			self._xtrain_var = self._xtrain.var(axis=0)
			with open(self._discr_info, "w") as f:
				f.write("Normalization mean".join(map(str,self._xtrain_mu)))
				f.write("Normalization var".join(map(str,self._xtrain_var)))
				f.write("\n")
			print("Mean:", self._xtrain_mu)
			print("Std:", np.sqrt(self._xtrain_var))
			self._xtest = self.MakeSamples(self._xtest_df)
		else:
			self._xtrain_df = None
			self._xtest_df = None
			self._xtrain = None
			self._xtest = None
			self._ytrain = None
			self._ytest = None
			self._xtrain_mu = None 
			self._xtrain_var = None 
		


	def VizInputs(self, label = "Training Samples"):
		indata = self._xtrain_df.drop(columns=["sample"])
		cols = self._features
		cols.append('label')
		indata = indata[cols]
		self.VizSamples(indata,label) 

	def VizTestSample(self):
		indata = self._xtest_df.drop(columns=["sample"])	
		cols = self._features
		cols.append('label')
		indata = indata[cols]
		self.VizSamples(indata,"Test Samples") 

	def StripLabels(self, indata):
		self._lb = LabelBinarizer()
		labels = indata["label"].to_numpy().reshape(-1,1)
		y = self._lb.fit_transform(labels)
		x = indata.drop("label",axis=1)
		return x, y
		

	def MakeSamples(self, indata):
		print("Samples are using features",self._features)
		x = indata[self._features].to_numpy()
		#normalizing data at keras level
		return x


	#fully connected network
	def BuildModel(self):
		#print("ytrain",self._ytrain[0])
		print("_xtest",self._xtest[0],"shape",self._xtest[0].shape)
		#need normalization layer
		norm = layers.Normalization(mean = self._xtrain_mu, variance = self._xtrain_var,name="normalization_layer")
		input_layer = Input(shape=self._xtest[0].shape)
		#reLu activation at internal layers
		dense_layers = [layers.Dense(n,name="dense_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]

		x_norm = norm(input_layer)
		x = dense_layers[0](x_norm)
		for i, d in enumerate(dense_layers[1:]):
			x = d(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(1,activation=activations.sigmoid,name="output")
		
		x = output_layer(x) 
		self._model = Model(inputs = input_layer, outputs = x, name = self._name)
		# check that normalization is correct
		x_train_norm = self._model.layers[1](self._xtrain).numpy()
		print("training norm mean",x_train_norm.mean(axis=0))
		print("training norm std",x_train_norm.std(axis=0))


	#SGD optimizer
	#cross-entropy loss (binary or categorical depending on labeling scheme)
	#monitor metrics: accuracy
	def CompileModel(self):
		self._model.compile(
			optimizer = 'adam',
			loss = 'binary_crossentropy',
			metrics = ['accuracy','AUC']
		)

	def MakePlots(self):
		self.VizSamples(self._xtest, self._ytest,"Training Samples")
	
	#data is a 1D array
	def MakeInputHist(self, data, col, label, fextra = ""):
		#scaler = MinMaxScaler()
		#data_norm = scaler.fit_transform(data)
		histdata = data[col]
		plotname = self._path+"/"+col
		if(fextra != ""):
			fextra = fextra.replace(" ","_")
			plotname += "_"+fextra
		if self._extra_label != "":
			plotname += "_"+self._extra_label
		plotname += "."+self._form
		bins = np.linspace(histdata.min(), histdata.max(), 50)
		ns, bins, _ = plt.hist(histdata,label=label,log=True,bins=bins,histtype=u'step')
		plt.title(col)
		plt.legend()
		print("Saving "+col+" "+label+" plot to",plotname)
		plt.savefig(plotname,format=self._form)
		plt.close()		

	def MakeMultiInputHist(self, indata, col, fextra = ""):
		plotname = self._path+"/"+col
		if(fextra != ""):
			fextra = fextra.replace(" ","_")
			plotname += "_"+fextra
		if self._extra_label != "":
			plotname += "_"+self._extra_label
		plotname += "."+self._form
		
		histmin = []
		histmax = []
		#print("catnames",catnames)
		for j, l in enumerate(self._catnames.keys()):
			mask = indata['label'] == l
			histdata = indata[mask][col]
			#clip data to be within pm 2std of mean
			histmean = np.mean(histdata)
			hist_std = np.std(histdata)
			histmin.append(max(histdata.min(), histmean - hist_std))
			histmax.append(min(histdata.max(), histmean + hist_std))
		bins = np.linspace(min(histmin), max(histmax), 50)
		for j, l in enumerate(self._catnames.keys()):
			mask = indata['label'] == l
			histdata = indata[mask][col].to_numpy()
			#mask array
			ns, bins, _ = plt.hist(histdata,label=self._catnames[l],log=True,bins=bins,histtype=u'step',color = self._catcolors[l])
		plottitle = col[col.find("Photon_")+7:]
		plottitle = plottitle[:plottitle.find("_CMS")]
		if plottitle.find("OvPhoton_Pt") != -1:
			plottitle = plottitle.replace("OvPhoton_Pt","/Pt")
		plt.title(plottitle)
		plt.legend()
		print("Saving "+col+" plot to",plotname)
		plt.savefig(plotname,format=self._form)
		plt.close()		
		
	
	def VizSamples(self, indata, extra = ""):
		cols = indata.columns
		for i, f in enumerate(cols):
			if f == "label":
				continue
			if 'idx' in f:
				continue
			#select column f with rows with label l
			self.MakeMultiInputHist(indata,f,extra)

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






