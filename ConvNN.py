from ModelBase import ModelBase
from tensorflow.keras import layers, metrics, Input, Model, activations, callbacks
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.preprocessing import normalize, MinMaxScaler, LabelBinarizer
import os
import subprocess
import numpy as np
from shap import DeepExplainer, summary_plot
from shap.plots import beeswarm
import matplotlib.pyplot as plt
import glob
from itertools import combinations

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



	def VizMetric(self, history, fname):
		plt.figure()
		plt.plot(history.history['val_'+fname], label="val "+fname)
		plt.plot(history.history[fname],label="train "+fname)
		plt.title(self._name+"\n"+fname)
		plt.legend()
		#check if output dir exists
		print("Saving loss plot to",self._path+"/"+fname+"."+self._form)
		plt.savefig(self._path+"/"+fname+"."+self._form,format=self._form)
		plt.close()


	#Caltech delayed photon analysis just plots fpr vs tpr for their DNN performance
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
			title=self._name+"\nSignal vs !signal subcluster ROC"
		)
		print("Saving ROC plot to",self._path+"/ROCplot."+self._form)
		plt.savefig(self._path+"/ROCplot."+self._form,format=self._form)
		plt.close()

	#ytrue and ypred are given in onehot form	
	#if cat = -1, plot one vs one for all classes
	#if cat != -1, plot cat vs all
	def VizMulticlassROC(self, ytrue, ypred, cat = -1):
		title=""

		ymin = 999
		fig = plt.figure()
		ax = plt.gca()
		#one vs all
		if cat != -1:
			catname = self._catnames[cat] 
			col = "pink" #also get from dict?
			title=self._name+"\n"+catname+" vs all subcluster ROC"
			fname = catname+"_vs_all"
			
			fpr, tpr, thresh = roc_curve(ytrue[:,cat], ypred[:, cat])
			
			#do 1- TPR
			tpr = [1 - i for i in tpr]
			if min(tpr[:-2]) < ymin:
				ymin = min(tpr[:-2])
			ax.plot(
				fpr,
				tpr,
				linewidth=4,
				label=catname+" vs rest",
				color=col,
			)
		#do all one vs ones
		else:
			#make unique pairs of categories
			ytrue_cat = self._lb.inverse_transform(ytrue)
			pairs = list(combinations(np.unique(ytrue_cat), 2))
			fname = "one_vs_ones"
			paircolors = {}
			for (cat1, cat2) in pairs:
				if self._catcolors[cat1] not in paircolors.values():
					paircolors[(cat1,cat2)] = self._catcolors[cat1]
				else:
					paircolors[(cat1,cat2)] = self._catcolors[cat2]
			for idx, (cat1, cat2) in enumerate(pairs):
				title=self._name+"\n"+self._catnames[cat1]+" vs "+self._catnames[cat2]+" subcluster ROC"
				#y_test needs to be categorical labels
				#print("cat1",cat1,self._catnames[cat1],"cat2",cat2,self._catnames[cat2])
				cat1_mask = ytrue_cat == cat1
				cat2_mask = ytrue_cat == cat2
				cat12_mask = np.logical_or(cat1_mask, cat2_mask)

				cat1_true = cat1_mask[cat12_mask]
				cat2_true = cat2_mask[cat12_mask]
							
				#get indices of categories in one-hot labels
				idx1 = np.flatnonzero(self._lb.classes_ == cat1)[0]
				idx2 = np.flatnonzero(self._lb.classes_ == cat2)[0]

				fpr_cat1, tpr_cat1, thresh_cat1 = roc_curve(cat1_true, ypred[cat12_mask, idx1])
				fpr_cat2, tpr_cat2, thresh_cat2 = roc_curve(cat2_true, ypred[cat12_mask, idx2])
				
				#do 1- TPR
				tpr_cat1 = [1 - i for i in tpr_cat1]
				if min(tpr_cat1[:-2]) < ymin:
					ymin = min(tpr_cat1[:-2])
				
				ax.plot(
					fpr_cat1,
					tpr_cat1,
					linewidth=4,
					label=self._catnames[cat1]+" vs "+self._catnames[cat2],
					color=paircolors[(cat1,cat2)],
				)
				'''
				RocCurveDisplay.from_predictions(
					cat1_true,
					ypred[cat12_mask, idx1],
					name=self._catnames[cat1]+" vs "+self._catnames[cat2],
					color=paircolors[(cat1,cat2)],
					ax=ax,
					pos_label = 1
    				)
				'''
			ax.legend()
		ax.set_yscale('log')	
		ax.grid()
		#focus on discriminating region of interest
		ax.set_ylim([1e-6,1.])
		#ax.set_ylim([1e-2, 5e-1])
		ax.set_xlim([0,0.1])
		ax.set(
			xlabel="FPR",
			ylabel="1 - TPR (misid rate)",
			title=title
		)
			#line, = ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
		print("Saving ROC plot to",self._path+"/ROC_"+fname+"."+self._form)
		plt.savefig(self._path+"/ROC_"+fname+"."+self._form,format=self._form)
		plt.close()

	def VizImportance(self):
		nsamp = 500
		#over all training data - should take a subset
		background = self._xtrain[np.random.choice(self._xtrain.shape[0], 100, replace=False)]
		vals = DeepExplainer(self._model, background).shap_values(self._xtrain[:nsamp])
		summary_plot(vals[0],self._xtrain[:nsamp],feature_names=self._features,show=False)
		print("Saving SHAP plot to",self._path+"/SHAPplot."+self._form)
		plt.savefig(self._path+"/SHAPplot."+self._form,format=self._form)
		plt.close()
		

	def TrainModel(self,epochs=1,batch=1000,viz=False,verb=1,savebest=False, earlystop=True):
		#remove old checkpoints in dir - update this to not use *	
		files = os.listdir(self._path)
		if any(".h5" in f for f in files):
			for file in glob.glob(self._path+"/*.h5"):
				os.remove(file)
			#subprocess.call("rm ./"+self._path+"/*.h5")
		#set checkpoint to save model with lowest validation loss (Caltech)
		callbacks_list = []
		if savebest:
			callback = callbacks.ModelCheckpoint(self._path+"/model_{epoch:03d}epoch_{val_loss:.5f}valloss.h5",monitor="val_loss",save_best_only=True,mode="min",initial_value_threshold=999.)
			callbacks_list.append(callback) 
		if earlystop:
			#do early stopping too
			earlystop_callback = callbacks.EarlyStopping("val_loss",min_delta=1e-6,mode='min',start_from_epoch=80)
			callbacks_list.append(earlystop_callback)
		#80/20 train/val split (of training data)
		#print("ytrain shape",self._ytrain.shape,np.array(self._ytrain).shape,np.array(self._ytrain)[0].shape,type(self._ytrain),type(np.array(self._xtrain)))
		his = self._model.fit(self._xtrain,self._ytrain,epochs=epochs,verbose=verb,validation_split=0.2,callbacks=callbacks_list,batch_size=batch)
		#save model with lowest validation loss
		if viz:
			self.VizMetric(his,"loss")

	def TestModel(self,batch_size=1,viz=False,verb=1,usebest=False):
		#get best model
		files = {}
		for root, dirs, f in os.walk(self._path):
			for name in f:
				if ".h5" not in name:
					continue
				valloss = name[name.rfind("_")+1:name.find("valloss")]
				files[valloss] = root+"/"+name
		keys = list(files.keys())
		
		#load best model
		print("loading model",files[min(keys)])	
		self._model.load_weights(files[min(keys)])	
		ypred = self._model.predict(self._xtest,batch_size=batch_size,verbose=verb)
		if viz:
			if len(self._ytest[0]) == 1:
				self.VizROC(self._ytest, ypred)
			else:  #multiclass
				#plot physics bkg vs other bkgs
				self.VizMulticlassROC(self._ytest, ypred,1)
				#plot one-v-one for each class
				self.VizMulticlassROC(self._ytest, ypred,-1)
			#self.VizImportance()
			#self.ValidateModel()

	'''
	#set to plot 1 entry (ie 1 grid) at a time
	def VizInputs(self):
		labels = self._lb.classes_
		all_labels = self._lb.inverse_transform(self._ytrain)
		inputs = [[[] for l in labels] for f in self._features]
		xtrain = self._scaler.inverse_transform(self._xtrain)
		for i, f in enumerate(inputs):
			self._inputHists.append([])
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
			print("Saving "+self._features[i]+" plot to",self._path+"/"+self._features[i]+"."+self._form)
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
				ns, bins, _ = ax1.hist(xtrain[:,i],label=self._catnames[labels[j]],weights=samp_weights[:,j],bins=self._inputHists[i][j]["bins"],histtype=u'step',color=self._catcolors[labels[j]])
				ax1.stairs(self._inputHists[i][j]["ns"],self._inputHists[i][j]["bins"],color=self._catcolors[labels[j]],fill=True,alpha=0.5)
				ax1.set_yscale('log')
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
			ax1.legend(handles = legend_elements_ax1)
			ax2.legend(handles = legend_elements_ax2)
			print("Saving predicted "+self._features[i]+" plot to",plotname)
			fig.savefig(plotname,format=self._form)
	'''
