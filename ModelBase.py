from abc import ABC, abstractmethod
import os
from shap import DeepExplainer, summary_plot
from shap.plots import beeswarm
import matplotlib.pyplot as plt
#from tensorflow.keras import layers, metrics, Input, Model, activations, callbacks
from sklearn.metrics import RocCurveDisplay, roc_curve
from tensorflow.keras import callbacks
import glob
from itertools import combinations
import numpy as np

class ModelBase(ABC):
	def __init__(self):
		self._model = None
		self._catnames = [] 
		self._catcolors = [] 
		super().__init__()
	
	@abstractmethod
	def BuildModel(self):
		pass

	@abstractmethod
	def CompileModel(self):
		pass

	def TrainModel(self,x,y,epochs = 1,viz:bool = False,verb= 0):
		his = self._model.fit(x,y,epochs=epochs,verbose=verb)
		if viz:
			VizLoss(his)

	def SetCategoryNames(self, catnames, catcolors = {}):
		self._catnames = catnames
		self._catcolors = catcolors
	
	def summary(self):
		self._model.summary()
	
	@abstractmethod
	def VizInputs(self):
		pass

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

	def VizImportance(self):
		nsamp = 500
		#over all training data - should take a subset
		background = self._xtrain[np.random.choice(self._xtrain.shape[0], 100, replace=False)]
		vals = DeepExplainer(self._model, background).shap_values(self._xtrain[:nsamp])
		summary_plot(vals[0],self._xtrain[:nsamp],feature_names=self._features,show=False)
		print("Saving SHAP plot to",self._path+"/SHAPplot."+self._form)
		plt.savefig(self._path+"/SHAPplot."+self._form,format=self._form)
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
	def VizMulticlassROC(self, ytrue, ypred, cat = -1, zoom = False):
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
			title = "one vs one ROC"
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
		if(zoom):
			ax.set_ylim([1e-2, 5e-1])
			ax.set_xlim([0,0.1])
			fname += "_zoom"
		else:
			ax.set_ylim([1e-6,1.])
			ax.set_xlim([0,1])
		ax.set(
			xlabel="FPR",
			ylabel="1 - TPR (misid rate)",
			title=title
		)
			#line, = ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
		print("Saving ROC plot to",self._path+"/ROC_"+fname+"."+self._form)
		plt.savefig(self._path+"/ROC_"+fname+"."+self._form,format=self._form)
		plt.close()

	#ytrue and ypred are given in onehot form	
	#if cat = -1, plot one vs one for all classes
	#if cat != -1, plot cat vs all
	def VizMulticlassROC(self, ytrue, ypred, cat = -1, zoom = False):
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
			title = "one vs one ROC"
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
		if(zoom):
			ax.set_ylim([1e-2, 5e-1])
			ax.set_xlim([0,0.1])
			fname += "_zoom"
		else:
			ax.set_ylim([1e-6,1.])
			ax.set_xlim([0,1])
		ax.set(
			xlabel="FPR",
			ylabel="1 - TPR (misid rate)",
			title=title
		)
			#line, = ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
		print("Saving ROC plot to",self._path+"/ROC_"+fname+"."+self._form)
		plt.savefig(self._path+"/ROC_"+fname+"."+self._form,format=self._form)
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
	
	def TestModel(self,batch_size=1,viz=False,verb=1,usebest=False, validate_model = False):
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
				self.VizMulticlassROC(self._ytest, ypred,-1,zoom=True)
			#self.VizImportance()
			if validate_model:
				self.ValidateModel()

	
	#load (input) weights
	def LoadWeights(self,checkpt_dir):
		print("Training network with latest weights from",checkpt_dir)
		latest = train.latest_checkpoint(checkpt_dir) 
		self._model.load_weights(latest)
		return
	
	#save (output) weights from current model
	def SaveWeights(self, batch_size, weights_dir):
		#save weights in nnue_training_weights
		checkpt_path = weights_dir+"/cp-{epoch:04d}.ckpt"
		checkpt_dir = os.path.dirname(checkpt_path)
		
		#create callback to save weights every epoch
		self.cp_callback = callbacks.ModelCheckpoint(
			filepath=checkpt_path,
			verbose=1,
			save_weights_only=True,
			save_freq = batch_size)
			#save_freq = 5*batch_size)
		self._model.save_weights(checkpt_path.format(epoch=0))
		print("Directory with model checkpoint is:",checkpt_dir)
		return checkpt_dir

	def GetModel(self):
		if self._model is None:
			print("Model not built yet")
			return
	def VizLoss(self, history, fname):
		plt.figure()
		plt.plot(history.history['val_loss'], label="val loss")
		plt.plot(history.history['loss'],label="train loss")
		plt.title("Loss")
		plt.legend()
		plt.savefig("loss/"+fname)



