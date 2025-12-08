from ModelBase import ModelBase
from keras import layers, metrics, Input, Model, activations
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler, LabelBinarizer
import os
import subprocess
import numpy as np
import mplhep as hep

class ConvNeuralNetwork(ModelBase):
	def __init__(self):
		self._inputShape = None
		self._nNodes = None
		self._xtrain = None
		self._ytrain = None
		self._xtrain_df = None
		self._ytrain_df = None
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
		self._tag = ""
		super().__init__()

	def __init__(self, data, nNodes, name = "model", tag = ""):
		self._bestModel = None
		self._wtrain = None
		self._lowestValLoss = 999
		self._form = "pdf"
		self._name = name
		if tag != "":
			self._name += "_"+tag
		self._path = "results/"+self._name
		self._catnames = {} 
		self._catcolors = {}
		self._inputHists = []
		self._tag = tag
		if not os.path.exists(self._path):
			os.mkdir(self._path)
		#a list of ints that defines the nodes for each dense layer (obviously len(nNodes) == # layers
		self._nNodes = nNodes
	
		y = self.StripLabels(data)
		#print("norm",x[:5],max(x[:,0]))
		#80/20 train/test split
		rand = 43 #change to random number to randomize
		self._xtrain_df, self._xtest_df, self._ytrain, self._ytest = train_test_split(data,y,test_size=0.2,random_state=rand)
		#rejoin labels for dfs
		ylabels = self._lb.inverse_transform(self._ytrain)
		self._xtrain_df["label"] = ylabels
		ylabels = self._lb.inverse_transform(self._ytest)
		self._xtest_df["label"] = ylabels
		
		#make grids for CNN inputs
		self._xtrain = self.MakeSamples(self._xtrain_df)
		self._xtest = self.MakeSamples(self._xtest_df)

		print("xtrain shape",self._xtrain.shape)

		self._ytrain = np.asarray([ np.asarray(i) for i in self._ytrain])
		#print(self._xtrain.shape[0],"training samples",self._ytrain.shape,type(self._ytrain),type(self._ytrain[0]),self._ytrain[0])
		#shape of input data
		super().__init__()

	def StripLabels(self, indata, labelcol = "label"):
		self._lb = LabelBinarizer()
		labels = indata[labelcol]
		y = self._lb.fit_transform(labels)
		indata.drop(labelcol,axis=1,inplace=True)
		return y

	def normalize_grids(self, grids):
		"""Normalize each grid individually. Modify as needed."""
		# Sum of energies per grid
		sums = grids.sum(axis=(1,2,3), keepdims=True)
		sums[sums == 0] = 1.0
		return grids / sums

	def MakeSamples(self, indata):
		# Preserve SC-level index
		sc_df = indata.reset_index(drop=True)
		sc_df["sc_id"] = np.arange(len(sc_df))

		# Explode to flat rechit rows
		col_eta = "SC_rh_iEta"
		col_phi = "SC_rh_iPhi"
		col_e = "SC_rh_Energy"
		if self._tag != "":
			col_eta = col_eta+"_"+self._tag
			col_phi = col_phi+"_"+self._tag
			col_e = col_e+"_"+self._tag
		exploded = sc_df.explode([col_eta, col_phi, col_e])

		# Extract numpy arrays
		sc_id  = exploded["sc_id"].to_numpy()
		iEta   = exploded[col_eta].to_numpy().astype(int)
		iPhi   = exploded[col_phi].to_numpy().astype(int)
		energy = exploded[col_e].to_numpy()

		# Convert local positions -3..3 → 0..6
		x = iEta + 3
		y = iPhi + 3

		# Allocate grids (N, 7, 7, 1)
		N = len(sc_df)
		grids = np.zeros((N, 7, 7, 1), dtype=np.float32)

		# Vectorized scatter
		grids[sc_id, x, y, 0] = energy

		#normalize grids
		grids = self.normalize_grids(grids)

		# Extract labels → (N, 1)
		return grids

		'''		
		#extract inputs and labels, remove unnecessary columns
		#drop event + subcl cols
		#dropcols = ["sample","event","object","label"]
		#x = data.drop(dropcols,axis=1)
		#if "subcl" in x.columns:
		#	x = data.drop(["subcl"],axis=1)
		

		#update processing for new CSV format (CSVs from root files)
		ngrid = 7
		half = ngrid // 2
	
		#assign sample index
		data["obj_idx"] = data.groupby(["event_idx","sc_idx"]).ngroup()
		nsamples = data["obj_idx"].nunique()

		#create input grid object
		X = np.zeros((nsamples, ngrid, ngrid, 1), dtype=np.float32)
		
		#get grid indices
		ix = (data["SC_rh_iEta_"+self._tag].values + half).astype(int)
		iy = (data["SC_rh_iPhi_"+self._tag].values + half).astype(int)
		si = data["obj_idx"].values

		#get channel value: energy * weight
		e = (data["SC_rh_Energy_"+self._tag] * data["SC_rh_Weight_"+self._tag]).values

		#safe insertion (handles duplicate rhs)
		np.add.at(X[:,:,:,0], (si, ix, iy), e)

		#labels	
		self._lb = LabelBinarizer()
		labels = data.groupby("obj_idx")["label"].first().values
		y = self._lb.fit_transform(labels)

		#normalization by sum of grid energy
		grids = X[:,:,:,0] #(N, 7, 7)
		totals = grids.sum(axis=(1,2), keepdims = True) + 1e-8
		X[:,:,:,0] = grids / totals
		'''
		return X, y	


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


	def TestModel(self,viz=True,batch_size=1,verb=1,validate_model = False, fpr_threshs = []):
		discr_threshs = super().TestModel(batch_size, verb, validate_model, fpr_threshs)
		#for testing with small dataset
		print("discr threshs",discr_threshs)
		if(viz):
			self.MakePlots(discr_threshs)

	def MakePlots(self, discr_threshs = []):
		#predicted test samples
		##take self._xtest_df and assign labels in a new column (labels) based on discriminator scores
		possible_labels = np.unique(self._lb.inverse_transform(self._ytest))
		score_cols = [f"score_{lab}" for lab in possible_labels]
		score_matrix = self._xtest_df[score_cols].to_numpy()
		mask = score_matrix > discr_threshs   # shape (nsamples, nclasses)
		# If no score passes threshold → return -1, assigns label to first class that passes discr_thresh
		pred_labels = np.where(mask.any(axis=1),
		                     possible_labels[mask.argmax(axis=1)],
		                     -1)


		self._xtest_df["pred_label"] = pred_labels
		pred_labels = self._lb.transform(pred_labels)
		self.VizSamples(self._xtest, pred_labels,"Test Sample Predictions")
		self.VizSamples(self._xtest, self._ytest,"Test Sample Truth")
		
		true_bh = self._xtrain_df[self._xtrain_df["label"] == 2]
		self.VizTimeVsEta(true_bh,"Training Sample True BH")
		pred_bh = self._xtest_df[self._xtest_df["pred_label"] == 2]
		self.VizTimeVsEta(pred_bh,"Test Sample Predicted BH")
		no_pred_bh = self._xtest_df[self._xtest_df["pred_label"] != 2]
		self.VizTimeVsEta(no_pred_bh,"Test Sample No Predicted BH")


		true_spike = self._xtest_df[self._xtest_df["label"] == 3]
		self.VizTimeVsEta(true_spike,"Test Sample True Spikes")
		pred_spike = self._xtest_df[self._xtest_df["pred_label"] == 3]
		self.VizTimeVsEta(pred_spike,"Test Sample Predicted Spikes")
		no_pred_spike = self._xtest_df[self._xtest_df["pred_label"] == 3]
		self.VizTimeVsEta(no_pred_spike,"Test Sample No Predicted Spikes")

		true_physbkg = self._xtest_df[self._xtest_df["label"] == 1]
		self.VizTimeVsEta(true_physbkg,"Test Sample True Physics Bkg")
		pred_physbkg = self._xtest_df[self._xtest_df["pred_label"] == 1]
		self.VizTimeVsEta(pred_physbkg,"Test Sample Predicted Physics Bkg")
		no_pred_physbkg = self._xtest_df[self._xtest_df["pred_label"] == 1]
		self.VizTimeVsEta(no_pred_physbkg,"Test Sample No Predicted Spikes")

	def VizInputs(self):
		self.VizSamples(self._xtrain, self._ytrain, "Training Samples")


	#can give self._xtrain_df or test version
	def VizTimeVsEta(self, data, extra_title):
		eta = data["SC_EtaCenter"]
		time = data["SC_seedTime"]

		plt.figure()
		ax = plt.gca()
		counts, xedges, yedges, im = plt.hist2d(time, eta, cmap='viridis',bins=50, range=[[-20,1],[-1.5,1.5]])
		cbar = plt.colorbar(im, ax=ax)
		#cbar.ax.tick_params(labelsize=10)
		cbar.set_label('a.u.')
		plt.xlabel("SC seed time [ns]")
		plt.ylabel("SC eta")
		hep.cms.label(llabel="Preliminary",com="13")
		plotname = self._path+"/"+"EtaVsTime"
		if(extra_title != ""):
			extra_title = extra_title.replace(" ","_")
			plotname += "_"+extra_title
		form = "png"
		plotname = plotname+"."+form
		print("Saving time vs eta distribution to",plotname)
		plt.savefig(plotname,format=form,dpi=500)
		plt.close()

	#set to plot 1 entry (ie 1 grid) at a time
	def VizSamples(self, indata, inlabels, extra_title = ""):
		labels = self._lb.inverse_transform(inlabels)
		labels_set = np.unique(labels)
		nlabels = len(labels_set)
		
		ncols = 2	
		nrows = int(np.ceil(nlabels / ncols))

		plt.figure(figsize = (5*ncols, 5*nrows-1))
		labels_dict  = {  1 : "physics bkg", 2 : "beam halo", 3 : "spikes"}

		for i, label in enumerate(labels_set):
			ax = plt.subplot(nrows, ncols, i+1)
			mask = (labels == label)
			cls_grid = indata[mask]
			if cls_grid.size == 0:
				continue
			avg_grid = cls_grid.mean(axis=0).squeeze()
			avg_grid /= avg_grid.sum()
			im = ax.imshow(avg_grid.T, origin='lower', cmap='viridis', interpolation='none', norm=colors.SymLogNorm(linthresh=1e-5, linscale=1,vmin=1e-4))
			ax.set_title(f"Average Grid ("+labels_dict[label]+")", fontsize=14)
			ax.set_xlabel("local iEta",fontsize=12)
			ax.set_ylabel("local iPhi",fontsize=12)
			ax.tick_params(axis="both",labelsize=10)
			# Annotate nonzero cells for clarity
			for r in range(7):
				for c in range(7):
					if avg_grid[r, c] != 0:
						ax.text(c, r, f"{avg_grid[r,c]:.2f}",
							ha="center", va="center", color="white", fontsize=7)


			cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
			cbar.ax.tick_params(labelsize=10)
			cbar.set_label('a.u.', fontsize=12)
		plt.subplots_adjust(hspace=0.1)
		plt.suptitle(self._name+"\n"+extra_title+" CNN Grids",fontsize=16)
		form = "png"
		plotname = self._path+"/"+"CNNInputGrids"
		if(extra_title != ""):
			extra_title = extra_title.replace(" ","_")
			plotname += "_"+extra_title
		plotname = plotname+"."+form
		print("Saving CNN grids to",plotname)
		plt.savefig(plotname,format=form,dpi=500)
		plt.close()
		#plt.show()

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
