from ModelBase import ModelBase
from tensorflow.keras import layers, metrics, Input, Model, activations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler, LabelBinarizer
import os
import subprocess
import numpy as np

class ConvNeuralNetwork(ModelBase):
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
		self._channels = []
		super().__init__()

	def __init__(self, data, nNodes, name = "model",channels = ["E","t","r"]):
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
		
		#print(x["CNNgrid_E_cell0_0"].head(),labels[:5])
		
		#drop not grid features
		gridcols = x.columns.str.contains("grid")
		grid = x.loc[:,gridcols]
		#print("grid cols",grid.columns,len(gridcols))
		#should be three channels so # cells is len cols/3
		ngrid = len(grid.columns)/3. #ngrid = 7x7
		ngrid = np.sqrt(ngrid) #ngrid = 7
		ngrid = int((ngrid-1)/2) #ngrid = 3
	
		self._scaler = [MinMaxScaler() for i in channels]
		list0 = []	
		##input to train_test_split is numpy array of samples, each sample is (7 x 7 x 3)	
		self._channels = channels
		for i in range(-ngrid,ngrid+1):
			cols_i = x.columns.str.contains("CNNgrid_E_cell"+str(i))
			grid_i = x.loc[:,gridcols]
			list_i = [] #list of cols to zip
			for j in range(-ngrid,ngrid+1):
				listcols = []
				col_E = x["CNNgrid_E_cell"+str(i)+"_"+str(j)]
				col_t = x["CNNgrid_t_cell"+str(i)+"_"+str(j)]
				col_r = x["CNNgrid_r_cell"+str(i)+"_"+str(j)]
				if "E" in channels:
					listcols.append(col_E)
				if "t" in channels:
					listcols.append(col_t)
				if "r" in channels:
					listcols.append(col_r)
				col = list(zip(*listcols))
				#print("i",i,"j",j,"total col",col[0])
				list_i.append(np.array(col))
				#list_i.append(x["CNNgrid_E_cell"+str(i)+"_"+str(j)])
			list_i = np.array(list_i)
			#print("list_"+str(i),list_i.shape)
			list_i = np.array(list(zip(*[l for l in list_i])))
			#print("zip list_"+str(i),list_i.shape)
			list0.append(list_i)
		x = np.array(list(zip(*[i for i in list0])))
		#print("x",x.shape,x[0])
		#print("channels",channels)
					
		self._features = []#x.columns
		#print("unnorm",x[0:5],max(x[:,0]))
	
		##normalize data - normalize each channel separately
		#print("energies",x0[0].flatten(),x0.flatten().shape)
		#print("first entry",x[0])	
		#do for each channel
		xnew = np.zeros(x.shape)
		for idx, scaler in enumerate(self._scaler):
			#print(channels[idx],x[:,:,:,idx].flatten(),x[:,:,:,idx].flatten().shape)
			xflat = [[i] for i in x[:,:,:,idx].flatten()]	
			scaler.fit(xflat)
			xnorm = scaler.transform(xflat).flatten()
			xnorm = xnorm.reshape((x.shape[0],x.shape[1],x.shape[2]))
			#print(x.shape,xnorm.shape)
			x[...,idx] = xnorm
		#print("normalized energies",x0[0].flatten(),x0.flatten().shape)	
		#print("normalized first entry",x[0])	
	
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
		print("xtrain",self._xtrain[0].shape)
		input_layer = Input(shape=self._xtrain[0].shape)
		#n filters with 5x5 kernels
		kernel_dim = 3
		#reLu activation at internal layers
		conv_layers = [layers.Conv2D(n,kernel_dim,name="conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x = conv_layers[0](input_layer)
		for i, c in enumerate(conv_layers[1:]):
			x = c(x)
		#flatten from 2D to 1D
		x = layers.Flatten()(x)
		x = layers.Dense(64,name="dense_layer",activation=activations.relu)(x)

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



	#set to plot 1 entry (ie 1 grid) at a time
	def VizInputs(self):
		labels = self._lb.classes_
		all_labels = self._lb.inverse_transform(self._ytrain)
		ngrid = self._xtrain.shape[1]
		hists2D = [[np.zeros((ngrid,ngrid)) for c in self._xtrain[0][0][0]] for l in labels] #hist needs x, y data
		for j, x in enumerate(self._xtrain):
			lidx = np.flatnonzero(self._lb.classes_ == all_labels[j])[0]
			for c, ch in enumerate(hists2D[lidx]):
				arr = x[:,:,c]
				hists2D[lidx][c] = np.sum([hists2D[lidx][c], arr],axis=0)
		for i, l in enumerate(labels):
			for c, ch in enumerate(hists2D[i]):
				#normalize histogram
				norm = sum(hists2D[i][c].flatten())
				hists2D[i][c] = hists2D[i][c]/norm
				hists2D[i][c] = hists2D[i][c].transpose()
				plotname = self._path+"/"+"CNNInputGrid_Label"+str(l)+"_Channel"+self._channels[c]+"."+self._form
				if os.path.exists(self._path+"/CNNInput_label"+str(l)+"_channel"+str(c)+"_"+self._name+"."+self._form):
					continue
				plt.title("Channel: "+self._channels[c]+" Label: "+self._catnames[l])
				plt.xlabel("local ieta")
				plt.ylabel("local iphi")
				plt.imshow(hists2D[i][c],extent=(-0.5 - (ngrid-1)/2, 0.5 + (ngrid-1)/2, -0.5 - (ngrid-1)/2, 0.5 + (ngrid-1)/2),origin="lower")
				plt.colorbar()
				print("Saving",plotname)
				plt.savefig(plotname,format=self._form)
				plt.close()		
