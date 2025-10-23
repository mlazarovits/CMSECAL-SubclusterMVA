from ModelBase import ModelBase
from keras import layers, metrics, Input, Model, activations
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
	
		x, y = self.ProcessData(data)
	
		#print("norm",x[:5],max(x[:,0]))
		#80/20 train/test split
		rand = 43 #change to random number to randomize
		self._xtrain, self._xtest, self._ytrain, self._ytest = train_test_split(x,y,test_size=0.2,random_state=rand)
		self._ytrain = np.asarray([ np.asarray(i) for i in self._ytrain])
		#print(self._xtrain.shape[0],"training samples",self._ytrain.shape,type(self._ytrain),type(self._ytrain[0]),self._ytrain[0])
		#shape of input data
		super().__init__()

	def ProcessData(self, data):
		self._lb = LabelBinarizer()
		labels = data["label"]
		y = self._lb.fit_transform(labels)
		
		#extract inputs and labels, remove unnecessary columns
		#drop event + subcl cols
		dropcols = ["sample","event","object","label"]
		x = data.drop(dropcols,axis=1)
		if "subcl" in x.columns:
			x = data.drop(["subcl"],axis=1)
		
		
		#drop not grid features
		gridcols = x.columns.str.contains("grid")
		grid = x.loc[:,gridcols]
		#print("grid cols",grid.columns,len(gridcols))
		ngrid = len(grid.columns) #ngrid = 7x7
		ngrid = np.sqrt(ngrid) #ngrid = 7
		ngrid = int((ngrid-1)/2) #ngrid = 3

	
		#do normalizations
		if "norm" in self._name:
			#get channel to normalize
			testname = self._name
			norm_cols = []
			for i in range(-ngrid,ngrid+1):
				for j in range(-ngrid,ngrid+1):
					norm_cols.append("CNNgrid_cell"+str(i)+"_"+str(j))
			sumcol = x[norm_cols].sum(axis=1)
			for i in range(-ngrid,ngrid+1):
				for j in range(-ngrid,ngrid+1):
					x["CNNgrid_cell"+str(i)+"_"+str(j)] = x["CNNgrid_cell"+str(i)+"_"+str(j)].div(sumcol)
	

		list0 = []	
		##input to train_test_split is numpy array of samples, each sample is (7 x 7 x nch)	
		for i in range(-ngrid,ngrid+1):
			cols_i = x.columns.str.contains("CNNgrid_cell"+str(i))
			grid_i = x.loc[:,gridcols]
			list_i = [] #list of cols to zip
			for j in range(-ngrid,ngrid+1):
				listcols = []
				#over all training samples
				#print("col_E",len(col_E),"multidx",multidx)
				col = x["CNNgrid_cell"+str(i)+"_"+str(j)]
				#listcols.append(col_E)
				#col = list(zip(*listcols))
				#print("col",col,"listcols",listcols)	
				#print("i",i,"j",j,"total col",col[0])
				list_i.append(np.array(col))
			list_i = np.array(list_i)
			#print("list_"+str(i),list_i.shape)
			list_i = np.array(list(zip(*[l for l in list_i])))
			#print("zip list_"+str(i),list_i.shape)
			list0.append(list_i)
		x = np.array(list(zip(*[i for i in list0]))) #should be size (nsamples, ngrid, ngrid, nchannels)
					
		self._features = []#x.columns
		#print("unnorm",x[0:5],max(x[:,0]))
	
		##normalize data - normalize each channel separately
		#print(channels[idx],x[:,:,:,idx].flatten(),x[:,:,:,idx].flatten().shape)
		scaler = MinMaxScaler()
		xflat = [[i] for i in x[:,:,:].flatten()]	
		scaler.fit(xflat)
		xnorm = scaler.transform(xflat).flatten()
		xnorm = xnorm.reshape((-1,x.shape[1],x.shape[2],1))
		#print(x.shape,xnorm.shape)
		x = xnorm
		return x, y	



	#convolutional network
	def BuildModel(self):
		input_layer = Input(shape=self._xtrain[0].shape)
		#n filters with 3x3 kernels
		kernel_dim = 3
		#reLu activation at internal layers
		conv_layers = [layers.Conv2D(n,kernel_dim,name="conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		#print("conv layer size",conv_layers[0].shape)
		x = conv_layers[0](input_layer)
		for i, c in enumerate(conv_layers[1:]):
			x = c(x)
		#flatten from 2D to 1D
		x = layers.Flatten()(x)
		x = layers.Dense(self._nNodes[-1]*2,name="dense_layer_1",activation=activations.relu)(x)
		x = layers.Dense(self._nNodes[-1]*2,name="dense_layer_2",activation=activations.relu)(x)

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
		hists2D = [np.zeros((ngrid,ngrid)) for l in labels] #hist needs x, y data
		for j, x in enumerate(self._xtrain):
			lidx = np.flatnonzero(self._lb.classes_ == all_labels[j])[0]
			arr = x[:,:,0] #take first channel (should only be 1)
			hists2D[lidx] = np.sum([hists2D[lidx], arr],axis=0)
		for i, l in enumerate(labels):
			#skip "mult" channels
			#normalize histogram
			norm = sum(hists2D[i].flatten())
			hists2D[i] = hists2D[i]/norm
			#put eta/phi on right axes
			hists2D[i] = hists2D[i].transpose()
			plotname = self._path+"/"+"CNNInputGrid_Label"+str(l)+"."+self._form
			if os.path.exists(self._path+"/CNNInput_label"+str(l)+"_"+self._name+"."+self._form):
				continue
			plt.title("Label: "+self._catnames[l])
			plt.xlabel("local ieta")
			plt.ylabel("local iphi")
			plt.imshow(hists2D[i],extent=(-0.5 - (ngrid-1)/2, 0.5 + (ngrid-1)/2, -0.5 - (ngrid-1)/2, 0.5 + (ngrid-1)/2),origin="lower")
			plt.colorbar()
			print("Saving",plotname)
			plt.savefig(plotname,format=self._form)
			plt.close()


	def VizModelWeights(self):
		#visualize filters (weights)
		#nNodes = list of length l for l layers, each entry is f filters
		for l, nf in enumerate(self._nNodes):
			print("layer",l+1,"has",nf,"filters")
			filters, biases = self._model.layers[l+1].get_weights()
			#normalize to [0,1]
			fmin, fmax = filters.min(), filters.max()
			filters = (filters - fmin)/(fmax - fmin)
			fig, axs = plt.subplots(nf,1,squeeze=True)
			plotname = self._model.layers[l+1].name+"_weights"
			fig.suptitle(plotname)
			plt.tight_layout()
			#plot each filter
			plt_idx = 0
			for f in range(nf):
				fil = filters[:,:,:,f]
				ax = axs[plt_idx]
				ax.set(title = "filter"+str(f))
				ax.set_xticks([])
				ax.set_yticks([])
				im = ax.imshow(fil[:,:])
				fig.colorbar(im, ax=ax, orientation='vertical')
				plt_idx += 1
			fig.subplots_adjust(left=0.4, right=0.6, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
			print("Saving",self._path+"/"+plotname+"."+self._form)
			plt.savefig(self._path+"/"+plotname+"."+self._form,format=self._form)
					 
	#visualize feature maps (weights applied)
	def VizFeatureMaps(self):
		#randomly select image from test dataset
		rng = np.random.default_rng(43)
		#randomly select index
		idxs = [i for i in range(len(self._xtest))]
		input_im_idx = rng.choice(idxs,1)
		input_im = self._xtest[input_im_idx]
		input_label = self._ytest[input_im_idx]
		input_label = self._lb.inverse_transform(input_label)[0]
		#input_im = self._xtest
		for l, nf in enumerate(self._nNodes):
			#create model from outputs of one layer
			model = Model(inputs = self._model.inputs,outputs = self._model.layers[l+1].output)
			feature_map = model.predict(input_im)
			#print("feature_map",feature_map.shape)
			#if only one pixel in feature map, skip
			if feature_map.shape[1] == 1 and feature_map.shape[0] == 1:
				continue 
			#will be nkernel feature maps per layer
			nkernel = self._model.layers[l+1].output.shape[-1]
			fig, axs = plt.subplots(nkernel,2)
			plotname = self._model.layers[l+1].name+"_featuremap"
			fig.suptitle(plotname)
			plt.tight_layout()
			for i in range(nkernel):
				ax = axs[i][0]
				ax.set(title = "kernel"+str(i))
				ax.set_xticks([])
				ax.set_yticks([])
				im = ax.imshow(feature_map[0, :, :, i].T)
				fig.colorbar(im, ax=ax, orientation='vertical')
			#plot input image
			ax_idx = int(nkernel/2)
			ax = axs[ax_idx][1]
			#turn off axes not used
			for i in range(nkernel):
				if i != ax_idx:
					axs[i][1].axis('off')
			ax.set(title = "input_image_"+self._catnames[input_label])
			ax.set_xlabel("local ieta")
			ax.set_ylabel("local iphi")
			ax.set_xticks([])
			ax.set_yticks([])
			im = ax.imshow(input_im[0,:,:,0].T)
			fig.colorbar(im, ax=ax, orientation='vertical')
			print("Saving",self._path+"/"+plotname+"."+self._form)
			plt.savefig(self._path+"/"+plotname+"."+self._form,format=self._form)
