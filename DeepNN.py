from ModelBase import ModelBase
from tensorflow.keras import layers, metrics, Input, Model, activations, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import normalize, MinMaxScaler, LabelBinarizer
import os
import subprocess
import numpy as np
from shap import DeepExplainer, summary_plot
from shap.plots import beeswarm
import matplotlib.pyplot as plt
import glob
from itertools import combinations

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
		super().__init__()

	def __init__(self, data, nNodes, name = "model"):
		self._bestModel = None
		self._lowestValLoss = 999
		self._form = "pdf"
		self._name = name
		self._path = "results/"+self._name
		self._catnames = {} 
		self._catcolors = {}
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
		self._xtrain, self._xtest, self._ytrain, self._ytrue = train_test_split(x,y,test_size=0.2,random_state=rand)
		self._ytrain = np.asarray([ np.asarray(i) for i in self._ytrain])
		#print(self._xtrain.shape[0],"training samples",self._ytrain.shape,type(self._ytrain),type(self._ytrain[0]),self._ytrain[0])
		#shape of input data
		super().__init__()

	def summary(self):
		self._model.summary()


	def SetCategoryNames(self, catnames, catcolors = {}):
		self._catnames = catnames
		self._catcolors = catcolors

	#fully connected network
	def BuildModel(self):
		input_layer = Input(shape=(self._xtrain.shape[1],))
		#reLu activation at internal layers
		dense_layers = [layers.Dense(n,name="dense_layer"+str(i),activation=activations.relu) for i, n in enumerate(self._nNodes)]
		x = dense_layers[0](input_layer)
		for i, d in enumerate(dense_layers[1:]):
			x = d(x)

		#sigmoid(binary)/softmax(multiclass) activation at the output layer to have interpretable probabilities
		output_layer = layers.Dense(len(self._ytrain[0]),activation=activations.sigmoid,name="output")
		
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

		#one vs all
		if cat != -1:
			catname = self._catnames[cat] 
			col = "pink" #also get from dict?
			title=self._name+"\n"+catname+" vs all subcluster ROC"
			fname = catname+"_vs_all"
			display = RocCurveDisplay.from_predictions(
				ytrue[:,cat],
				ypred[:,cat],
				name=catname+" vs rest",
				color="pink",
				plot_chance_level=True,
			)
			display.ax_.set(
				xlabel="False Positive Rate",
				ylabel="True Positive Rate",
				title=title
			)
		#do all one vs ones
		else:
			#make unique pairs of categories
			ytrue_cat = self._lb.inverse_transform(ytrue)
			pairs = list(combinations(np.unique(ytrue_cat), 2))
			fname = "one_vs_ones"
			fig = plt.figure()
			ax = plt.gca()
			ax.set(
				xlabel="False Positive Rate",
				ylabel="True Positive Rate",
				title=title
			)
			paircolors = {}
			for (cat1, cat2) in pairs:
				if self._catcolors[cat1] not in paircolors.values():
					paircolors[(cat1,cat2)] = self._catcolors[cat1]
				else:
					paircolors[(cat1,cat2)] = self._catcolors[cat2]
			for idx, (cat1, cat2) in enumerate(pairs):
				title=self._name+"\n"+self._catnames[cat1]+" vs "+self._catnames[cat2]+" subcluster ROC"
				#y_test needs to be categorical labels
				
				cat1_mask = ytrue_cat == cat1
				cat2_mask = ytrue_cat == cat2
				cat12_mask = np.logical_or(cat1_mask, cat2_mask)

				cat1_true = cat1_mask[cat12_mask]
				cat2_true = cat2_mask[cat12_mask]
							
				#get indices of categories in one-hot labels
				idx1 = np.flatnonzero(self._lb.classes_ == cat1)[0]
				idx2 = np.flatnonzero(self._lb.classes_ == cat2)[0]

				RocCurveDisplay.from_predictions(
					cat1_true,
					ypred[cat12_mask, idx1],
					name=self._catnames[cat1]+" vs "+self._catnames[cat2],
					color=paircolors[(cat1,cat2)],
					ax=ax
    				)
			line, = ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
			#h, l = ax.get_legend_handles_labels()
			#h.append(line)
			ax.legend()
		print("Saving ROC plot to",self._path+"/ROCplot_"+fname+"."+self._form)
		plt.savefig(self._path+"/ROCplot"+fname+"."+self._form,format=self._form)
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
			
	def VizInputs(self):
		labels = self._lb.classes_
		all_labels = self._lb.inverse_transform(self._ytrain)
		inputs = [[[] for l in labels] for f in self._features]
		xtrain = self._scaler.inverse_transform(self._xtrain)
		for i, f in enumerate(inputs):
			if os.path.exists(self._path+"/"+self._features[i]+"."+self._form):
				continue
			for j, x in enumerate(xtrain):
				#print(i,x,self._ytrain[j],self._features[i])
				#this sample needs to be put in j == label[k]
				#print(self._xtrain[j],self._ytrain[j],all_labels[j])
				lidx = np.flatnonzero(self._lb.classes_ == all_labels[j])[0]
				#print(i,self._features[i])
				inputs[i][lidx].append(x[i])
			for j, l in enumerate(labels):
				plt.hist(inputs[i][j],label=self._catnames[labels[j]],log=True,bins=50,histtype=u'step')
			#plt.hist(x[1],label="sig",histtype=u'step',log=True,bins=50)
			plt.title(self._features[i])
			plt.legend()
			print("Saving "+self._features[i]+" plot to",self._path+"/"+self._features[i]+"."+self._form)
			plt.savefig(self._path+"/"+self._features[i]+"."+self._form,format=self._form)
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
			#can also add accuracy, etc.

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
			if len(self._ytrue[0]) == 1:
				self.VizROC(self._ytrue, ypred)
			else:  #multiclass
				#plot physics bkg vs other bkgs
				self.VizMulticlassROC(self._ytrue, ypred,1)
				#plot one-v-one for each class
				self.VizMulticlassROC(self._ytrue, ypred,-1)
			#self.VizImportance()	
