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
	
		#do normalizations
		if "norm" in self._name:
			#get channel to normalize
			testname = self._name
			norm_chs = []
			while testname.count("norm") > 0:
				match_idx = testname.find("norm")+4
				normCh = testname[match_idx:match_idx+1]
				norm_chs.append(normCh) 
				testname = testname[match_idx+2:]
			for ch in norm_chs:
				print("normalizing channel:",ch)
				norm_cols = []
				for i in range(-ngrid,ngrid+1):
					for j in range(-ngrid,ngrid+1):
						norm_cols.append("CNNgrid_"+ch+"_cell"+str(i)+"_"+str(j))
				sumcol = x[norm_cols].sum(axis=1)
				for i in range(-ngrid,ngrid+1):
					for j in range(-ngrid,ngrid+1):
						x["CNNgrid_"+ch+"_cell"+str(i)+"_"+str(j)] = x["CNNgrid_"+ch+"_cell"+str(i)+"_"+str(j)].div(sumcol)
	


		self._scaler = [MinMaxScaler() for i in channels]
		list0 = []	
		##input to train_test_split is numpy array of samples, each sample is (7 x 7 x nch)	
		self._channels = channels
		multidx = self._name.find("Mult")
		for i in range(-ngrid,ngrid+1):
			cols_i = x.columns.str.contains("CNNgrid_E_cell"+str(i))
			grid_i = x.loc[:,gridcols]
			list_i = [] #list of cols to zip
			for j in range(-ngrid,ngrid+1):
				listcols = []
				#over all training samples
				col_E = x["CNNgrid_E_cell"+str(i)+"_"+str(j)]
				col_t = x["CNNgrid_t_cell"+str(i)+"_"+str(j)]
				col_r = x["CNNgrid_r_cell"+str(i)+"_"+str(j)]
				if multidx != -1:
					#get channels that are multiplied together
					multchs = [self._name[multidx+4],self._name[multidx-1]]
					if ("E" in multchs) and ("r" in multchs):
						col_Er = col_E.mul(col_r)
						#locs = col_r.loc[(col_r < 1) & (col_r > 0)].index.tolist()
						#print("colE",col_E,"colr",col_r,"colEr",col_Er)
						listcols.append(col_Er)
					if ("E" in multchs) and ("t" in multchs):
						col_Et = col_E.mul(col_t) 
						listcols.append(col_Et)
					if ("t" in multchs) and ("r" in multchs):
						col_tr = col_t.mul(col_r) 
						listcols.append(col_tr)
				else:
					if "E" in channels:
						listcols.append(col_E)
					if "t" in channels:
						listcols.append(col_t)
					if "r" in channels:
						listcols.append(col_r)
				col = list(zip(*listcols))
				#print("col",col,"listcols",listcols)	
				#print("i",i,"j",j,"total col",col[0])
				list_i.append(np.array(col))
				#list_i.append(x["CNNgrid_E_cell"+str(i)+"_"+str(j)])
			list_i = np.array(list_i)
			#print("list_"+str(i),list_i.shape)
			list_i = np.array(list(zip(*[l for l in list_i])))
			#print("zip list_"+str(i),list_i.shape)
			list0.append(list_i)
		x = np.array(list(zip(*[i for i in list0]))) #should be size (nsamples, ngrid, ngrid, nchannels)
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

	#convolutional network
	def BuildModel(self):
		input_layer = Input(shape=self._xtrain[0].shape)
		#n filters with 3x3 kernels
		kernel_dim = 3
		#reLu activation at internal layers
		conv_layers = [layers.Conv2D(n,kernel_dim,name="conv_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
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
		hists2D = [[np.zeros((ngrid,ngrid)) for c in self._channels] for l in labels] #hist needs x, y data
		#print("len hists2D 0",len(hists2D[0]),"len chn",len(self._channels),"xtrain000",self._xtrain[0][0][0])
		for j, x in enumerate(self._xtrain):
			lidx = np.flatnonzero(self._lb.classes_ == all_labels[j])[0]
			for c, ch in enumerate(hists2D[lidx]):
				arr = x[:,:,c]
				hists2D[lidx][c] = np.sum([hists2D[lidx][c], arr],axis=0)
		for i, l in enumerate(labels):
			for c, ch in enumerate(hists2D[i]):
				#skip "mult" channels
				if "Mult" in self._channels[c]:
					continue
				#normalize histogram
				norm = sum(hists2D[i][c].flatten())
				hists2D[i][c] = hists2D[i][c]/norm
				hists2D[i][c] = hists2D[i][c].transpose()
				plotname = self._path+"/"+"CNNInputGrid_Label"+str(l)+"_Channel"+self._channels[c]+"."+self._form
				if os.path.exists(self._path+"/CNNInput_label"+str(l)+"_channel"+str(c)+"_"+self._name+"."+self._form):
					continue
				print("c",c,"channel",self._channels[c])
				plt.title("Channel: "+self._channels[c]+" Label: "+self._catnames[l])
				plt.xlabel("local ieta")
				plt.ylabel("local iphi")
				plt.imshow(hists2D[i][c],extent=(-0.5 - (ngrid-1)/2, 0.5 + (ngrid-1)/2, -0.5 - (ngrid-1)/2, 0.5 + (ngrid-1)/2),origin="lower")
				plt.colorbar()
				print("Saving",plotname)
				plt.savefig(plotname,format=self._form)
				plt.close()


	def VizModelWeights(self):
		#visualize filters (weights)
		#nNodes = list of length l for l layers, each entry is f filters
		nCh = len(self._channels)
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
				#plot each channel
				for c, ch in enumerate(self._channels):
					ax = axs[plt_idx]
					ax.set(title = "filter"+str(f)+"_ch"+ch)
					ax.set_xticks([])
					ax.set_yticks([])
					im = ax.imshow(fil[:,:,c])
					fig.colorbar(im, ax=ax, orientation='vertical')
					plt_idx += 1
			fig.subplots_adjust(left=0.4, right=0.6, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
			print("Saving",self._path+"/"+plotname+"."+self._form)
			plt.savefig(self._path+"/"+plotname+"."+self._form,format=self._form)
					 
	#visualize feature maps (weights applied)
	def VizFeatureMaps(self):
		#randomly select image from test dataset
		rng = np.random.default_rng(45)
		#randomly select index
		idxs = [i for i in range(len(self._xtest))]
		input_im_idx = rng.choice(idxs,1)
		input_im = self._xtest[input_im_idx]
		input_label = self._ytest[input_im_idx]
		input_label = self._lb.inverse_transform(input_label)[0]
		#input_im = self._xtest
		nCh = len(self._channels)
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
				im = ax.imshow(feature_map[0, :, :, i])
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
			im = ax.imshow(input_im[0])
			fig.colorbar(im, ax=ax, orientation='vertical')
			print("Saving",self._path+"/"+plotname+"."+self._form)
			plt.savefig(self._path+"/"+plotname+"."+self._form,format=self._form)
