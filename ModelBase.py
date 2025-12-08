from abc import ABC, abstractmethod
import os
#from shap import DeepExplainer, summary_plot
#from shap.plots import beeswarm
import matplotlib.pyplot as plt
#from tensorflow.keras import layers, metrics, Input, Model, activations, callbacks
from sklearn.metrics import RocCurveDisplay, roc_curve
from keras import callbacks
import glob
from itertools import combinations
import numpy as np
import pandas as pd
import mplhep as hep
import dask.array as da
import dask.dataframe as dd


class ModelBase(ABC):
	def __init__(self):
		hep.style.use("CMS")
		self._model = None
		self._catnames = [] 
		self._catcolors = []
		self._wtrain = None
		self._train_gen = None
		self._val_gen = None
		self._test_gen = None
		super().__init__()
	
	@abstractmethod
	def BuildModel(self):
		pass

	@abstractmethod
	def CompileModel(self):
		pass

	@abstractmethod
	def StripLabels(self, indata):
		pass
	
	@abstractmethod
	def MakeSamples(self, indata):
		pass

	def SetCategoryNames(self, catnames, catcolors = {}):
		self._catnames = catnames
		self._catcolors = catcolors
	
	def summary(self):
		self._model.summary()
	
	@abstractmethod
	def VizSamples(self):
		pass

	def VizMetric(self, history, fname):
		plt.figure()
		plt.plot(history.history['val_'+fname], label="val "+fname)
		plt.plot(history.history[fname],label="train "+fname)
		plt.title(self._name+"\n"+fname,fontsize=10)
		plt.legend()
		#check if output dir exists
		print("Saving loss plot to",self._path+"/"+fname+"."+self._form)
		plt.savefig(self._path+"/"+fname+"."+self._form,format=self._form)
		plt.close()

	def VizImportance(self):
		nsamp = 500
		##over all training data - should take a subset
		#background = self._xtrain[np.random.choice(self._xtrain.shape[0], 100, replace=False)]
		#vals = DeepExplainer(self._model, background).shap_values(self._xtrain[:nsamp])
		#summary_plot(vals[0],self._xtrain[:nsamp],feature_names=self._features,show=False)
		#print("Saving SHAP plot to",self._path+"/SHAPplot."+self._form)
		#plt.savefig(self._path+"/SHAPplot."+self._form,format=self._form)
		#plt.close()

	def FindDiscThresh(self, fpr_thresh, ncat, fpr_cat, tpr_cat, thresh_cat):
		mindiff = 999
		bestIdx = 0
		for i, fpr in enumerate(fpr_cat):
			diff = abs(fpr - fpr_thresh)
			if diff < mindiff:
				mindiff = diff
				bestIdx = i
		print("FPR ~"+str(fpr_thresh)+", cat (sig)",ncat,self._catnames[ncat],"fpr",fpr_cat[bestIdx],"tpr",tpr_cat[bestIdx],"thresh on sig cat",thresh_cat[bestIdx])
		return thresh_cat[bestIdx]

	def MakeROC(self, ytrue, ypred, pos_label=1, fpr_threshs = [], ret_fpr_thresh = -1):
		print("pos_label",pos_label,"# ytrue",len(ytrue),"# ypred",len(ypred),"ytrue",ytrue[0],"ypred",ypred[0])
		#need to process ytrue and ypred s.t. they are given to roc_curve as 1D arrays of assignment (ytrue - 0 or 1) and prediction (score of 'signal'/positive class)
		ytrue_1D = []
		ypred_1D = []
		for y in range(len(ytrue)):
			ytrue_1D.append(ytrue[y][pos_label])
			ypred_1D.append(ypred[y][pos_label])
		#dont need to give 'pos label' to roc_curve since those values have been selected above
		fpr, tpr, thresh = roc_curve(ytrue_1D, ypred_1D)

		ytrue_test = np.zeros(ytrue[0].shape)
		#put in one-hot encoding
		pos_cat = [1 if idx == pos_label else 0 for idx, i in enumerate(ytrue_test)]
		neg_cat = [1 if idx != pos_label else 0 for idx, i in enumerate(ytrue_test)]
		poscat = self._lb.inverse_transform([pos_cat])[0][0]
		for fpr_thresh in fpr_threshs:
			self.FindDiscThresh(fpr_thresh, poscat, fpr, tpr, thresh)
		if ret_fpr_thresh != -1:
			discr_thresh = self.FindDiscThresh(ret_fpr_thresh, poscat, fpr, tpr, thresh)
	
		#tpr = signal efficiency
		#1 - tpr = fnr = signal inefficiency
		#fpr = background mistag rate
		#1 - fpr = tnr = background rejection

		#TODO - make sure ROC curve is in best variables and log scales, ranges, etc to see what's going on the best	
		#do 1 - FPR = TNR
		#fpr = [1 - i for i in fpr]
		#do 1 - TPR = FNR
		#tpr = [1 - i for i in tpr]
		return fpr, tpr, ret_discr_thresh

	def PlotROCs(self, fprs, tprs, labels, colors = [], fextra = "", class1name = "sig", class2name = "bkg"):
		fig = plt.figure()
		ax = plt.gca()
	
		#tpr = signal efficiency
		#1 - tpr = fnr = signal inefficiency
		#fpr = background mistag rate
		#1 - fpr = tnr = background rejection

		#TODO - make sure ROC curve is in best variables and log scales, ranges, etc to see what's going on the best	
		#do 1 - FPR = TNR
		#fpr = [1 - i for i in fpr]
		#do 1 - TPR = FNR
		#tpr = [1 - i for i in tpr]
		for fpr, tpr, l, col in zip(fprs, tprs, labels, colors):
			ax.plot(
				fpr,
				tpr,
				linewidth=4,
				label=l,
				color=col,
			)
		ax.set(
			xlabel="Background mistag",
			ylabel="Signal efficiency",
		)
		ax.set_title(self._name+"\n"+class1name+" (sig) vs "+class2name+" (bkg) ROC",fontsize=16)
		ax.set_ylim([0.8, 1.0])
		ax.set_xlim([1e-6,0.3])
		ax.grid()
		if(labels != [""]):
			ax.legend()

		plotname = self._path+"/ROCplot"
		if fextra != "":
			plotname += "_"+fextra
		plotname += "."+self._form 
		print("Saving ROC plot to",plotname)
		plt.savefig(plotname,format=self._form)
		plt.close()

	

	#Caltech delayed photon analysis just plots fpr vs tpr for their DNN performance
	#for multiclass ROC (one-vs-rest = sig-vs-rest)
	def VizROC(self, ytrue, ypred, class1name = "sig", class2name = "bkg", pos_label=1, fextra="", fpr_threshs = [], fpr_thresh = -1):
		fpr, tpr, discr_thresh = self.MakeROC(ytrue, ypred, pos_label, fpr_threshs)
		self.PlotROCs([fpr.tolist()], [tpr.tolist()], [""],["pink"], fextra, class1name, class2name)
		return discr_thresh
	
	#ytrue and ypred are given in onehot form	
	#if cat = -1, plot one vs one for all classes
	#if cat != -1, plot cat vs all
	def VizMulticlassROC(self, ytrue, ypred, cat = -1, zoom = False, fextra = "", fpr_thresh = -1):
		title=""

		fig = plt.figure()
		ax = plt.gca()
		#one vs all
		if cat != -1:
			catname = self._catnames[cat] 
			col = "pink" #also get from dict?
			title=self._name+"\n"+catname+" vs all ROC"
			fname = catname+"_vs_all"
			ncat = self._lb.transform([cat])[0]
			ncat = np.where(ncat == 1)[0]
	
			fpr, tpr, thresh = roc_curve(ytrue[:,ncat], ypred[:,ncat])
			print("cat",catname,"vs all")
			self.FindDiscThresh(0.02, cat, fpr, tpr, thresh)
			self.FindDiscThresh(0.01, cat, fpr, tpr, thresh)
			self.FindDiscThresh(0.005, cat, fpr, tpr, thresh)
			self.FindDiscThresh(0.001, cat, fpr, tpr, thresh)
			discr_thresh = self.FindDiscThresh(fpr_thresh, cat, fpr, tpr, thresh)
			print("discr_thresh",discr_thresh)		
			#do 1- TPR
			#tpr = [1 - i for i in tpr]
			ax.plot(
				fpr,
				tpr,
				linewidth=4,
				label=catname+" vs rest",
				color=col,
			)
		#do all one vs ones
		else:
			title=self._name+"\n 1-v-1 ROC"
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
				#only do for bkg classes vs phys bkg
				#if(cat1 != 1 and cat2 != 1):
				#	continue

				#y_test needs to be categorical labels
				#only focus on the 2 categories under analysis rn - hence logical or
				cat1_mask = ytrue_cat == cat1
				cat2_mask = ytrue_cat == cat2
				cat12_mask = np.logical_or(cat1_mask, cat2_mask)

				cat1_true = cat1_mask[cat12_mask]
				cat2_true = cat2_mask[cat12_mask]
						
				#get indices of categories in one-hot labels
				idx1 = np.flatnonzero(self._lb.classes_ == cat1)[0]
				idx2 = np.flatnonzero(self._lb.classes_ == cat2)[0]

				print("cat1",cat1,"cat2",cat2)
				print("ytrue",ytrue[0],"ypred",ypred[0])
				print("cat1_true",cat1_true[0],"ypred",ypred[cat12_mask, idx1][0])
				print("cat2_true",cat2_true[0],"ypred",ypred[cat12_mask, idx2][0])

				#whichever true cat is given is 'positive' class
				fpr_cat1, tpr_cat1, thresh_cat1 = roc_curve(cat1_true, ypred[cat12_mask, idx1])
				fpr_cat2, tpr_cat2, thresh_cat2 = roc_curve(cat2_true, ypred[cat12_mask, idx2])
				
				#TODO - write out fpr, tprs to csv

				##do 1- TPR
				#tpr_cat1 = [1 - i for i in tpr_cat1]
				#print("fpr, tpr, thresh for cat 2",list(zip(fpr_cat1, tpr_cat1, thresh_cat2)))
				#print("fpr, tpr, thresh for cat 1",list(zip(fpr_cat1, tpr_cat1, thresh_cat1)))

				#get FPR (misid) ~ 0.02
				#find index of entry in tpr for element that is closest to 0.02
				self.FindDiscThresh(0.02, cat1, fpr_cat1, tpr_cat1, thresh_cat1)
				self.FindDiscThresh(0.02, cat2, fpr_cat2, tpr_cat2, thresh_cat2)
				self.FindDiscThresh(0.01, cat1, fpr_cat1, tpr_cat1, thresh_cat1)
				self.FindDiscThresh(0.01, cat2, fpr_cat2, tpr_cat2, thresh_cat2)
				self.FindDiscThresh(0.005, cat1, fpr_cat1, tpr_cat1, thresh_cat1)
				self.FindDiscThresh(0.005, cat2, fpr_cat2, tpr_cat2, thresh_cat2)
				self.FindDiscThresh(0.001, cat1, fpr_cat1, tpr_cat1, thresh_cat1)
				self.FindDiscThresh(0.001, cat2, fpr_cat2, tpr_cat2, thresh_cat2)
					
				discr_thresh = -1
				#mindiff = 999
				#bestIdx = 0
				#for i, fpr in enumerate(fpr_cat2):
				#	diff = abs(fpr - 0.02)
				#	if diff < mindiff:
				#		mindiff = diff
				#		bestIdx = i
				#print("FPR ~ 2% cat2 (sig)",cat2,self._catnames[cat2],"fpr",fpr_cat2[bestIdx],"tpr",tpr_cat2[bestIdx],"thresh cat2",thresh_cat2[bestIdx])
					
				#use cat1 bc this is always label 1 (phys bkg) based on how the classes were paired
				#so the positive label will be 1 (ie if something is classified as phys bkg, it should have ypred = [1, 0, 0] instead of ie [0, 1, 0] for cat 1 vs cat2)
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
		#ax.set_yscale('log')	
		ax.grid()
		#focus on discriminating region of interest
		if(zoom):
			ymin = 7e-1
			xmax = 0.1
			fname += "_zoom"
		else:
			ymin = 1e-6
			xmax = 1.
		ax.set_ylim([ymin, 1.])
		ax.set_yticks(np.arange(ymin,1,0.05))
		ax.set_xlim([0,xmax])
		ax.set(
			xlabel="background-as-signal mistag rate", #FPR
			ylabel="signal efficiency", #1 - TPR
		)
		ax.set_title(title,fontsize=16)
		plotname = self._path+"/ROC_"+fname
		if fextra != "":
			plotname += "_"+fextra
		plotname += "."+self._form 
		print("Saving ROC plot to",plotname)
		plt.savefig(plotname,format=self._form)
		plt.close()
		return discr_thresh


	def TrainModel(self,epochs=1,batch=1000,viz=False,verb=1,savebest=False, earlystop=True):
		#remove old checkpoints in dir - update this to not use *	
		files = os.listdir(self._path)
		if any(".keras" in f for f in files):
			for file in glob.glob(self._path+"/*.keras"):
				os.remove(file)
			#subprocess.call("rm ./"+self._path+"/*.keras")
		#set checkpoint to save model with lowest validation loss (Caltech)
		callbacks_list = []
		if savebest:
			callback = callbacks.ModelCheckpoint(self._path+"/model_{epoch:03d}epoch_{val_loss:.5f}valloss.keras",monitor="val_loss",save_best_only=True,mode="min",initial_value_threshold=999.)
			callbacks_list.append(callback) 
		if earlystop:
			#do early stopping too
			earlystop_callback = callbacks.EarlyStopping("val_loss",min_delta=1e-6,mode='min',start_from_epoch=80)
			callbacks_list.append(earlystop_callback)
		#80/20 train/val split (of training data)
		#print("ytrain shape",self._ytrain.shape,np.array(self._ytrain).shape,np.array(self._ytrain)[0].shape,type(self._ytrain),type(np.array(self._xtrain)))
		if self._wtrain is None:
			his = self._model.fit(self._xtrain,self._ytrain,epochs=epochs,verbose=verb,validation_split=0.2,callbacks=callbacks_list,batch_size=batch)
		else:
			his = self._model.fit(self._xtrain,self._ytrain,sample_weight=self._wtrain,epochs=epochs,verbose=verb,validation_split=0.2,callbacks=callbacks_list,batch_size=batch)
		#save model with lowest validation loss
		if viz:
			self.VizMetric(his,"loss")
		#print("cats",self._catnames)


	def TrainModelGenerator(self,epochs=1,batch=1000,viz=False,verb=1,savebest=False, earlystop=True):
		#remove old checkpoints in dir - update this to not use *	
		files = os.listdir(self._path)
		if any(".keras" in f for f in files):
			for file in glob.glob(self._path+"/*.keras"):
				os.remove(file)
			#subprocess.call("rm ./"+self._path+"/*.keras")
		#set checkpoint to save model with lowest validation loss (Caltech)
		callbacks_list = []
		if savebest:
			callback = callbacks.ModelCheckpoint(self._path+"/model_{epoch:03d}epoch_{val_loss:.5f}valloss.keras",monitor="val_loss",save_best_only=True,mode="min",initial_value_threshold=999.)
			callbacks_list.append(callback) 
		if earlystop:
			#do early stopping too
			earlystop_callback = callbacks.EarlyStopping("val_loss",min_delta=1e-6,mode='min',start_from_epoch=80)
			callbacks_list.append(earlystop_callback)
		#80/20 train/val split (of training data)
		#print("ytrain shape",self._ytrain.shape,np.array(self._ytrain).shape,np.array(self._ytrain)[0].shape,type(self._ytrain),type(np.array(self._xtrain)))
		his = self._model.fit(self._train_gen,epochs=epochs,verbose=verb,validation_split=0.2,callbacks=callbacks_list,batch_size=batch)
		#save model with lowest validation loss
		if viz:
			self.VizMetric(his,"loss")
		#print("cats",self._catnames)


	def LoadBestModel(self):
		if not os.path.exists(self._path):
			print("Error: network at",self._path,"does not exist. Select another network or train this one.")
			return
		#get best model
		files = {}
		for root, dirs, f in os.walk(self._path):
			for name in f:
				if ".keras" not in name:
					continue
				valloss = name[name.rfind("_")+1:name.find("valloss")]
				files[valloss] = root+"/"+name
		if(len(files) < 1):
			print("No models found.")
			return
		keys = list(files.keys())
		
		#load best model
		print("Loading model",files[min(keys)])
		self._model.load_weights(files[min(keys)])	

	def TestModel_EnergySplit(self,x,energy,ytrue,ypred,pos_label,batch_size=1,verb=1,fextra=""):
		#do preprocessing for energy-separated roc curves
		energy_ranges = []
		nclasses = len(ypred[0])
		energy_ranges = [[30, 75],[75,150], [150, 200], [200]]
		#TODO: change to color gradient (low to high energy)
		colors = ["blue","green","purple","pink"]
		print("Creating energy-separate ROC curves with energy bins",energy_ranges)
		#create dataframe of xtest, ytrue, ypred
		df_e = pd.DataFrame()
		#print("energy",energy[0],"ypred",ypred[0],'ytest',ytrue[0])
		df_e['energy'] = energy 
		df_e['ypred'] = ypred.tolist()
		df_e['ytrue'] = ytrue.tolist()
		#print("energy split - labels",np.unique(df_e['ytrue'].to_numpy()))
		mask_df_2 =df_e['ytrue'].apply(lambda x: x == [0, 1])
		#do energy breakdown
		fprs = []
		tprs = []
		labels = []
		for idx, erange in enumerate(energy_ranges):
			mask = None
			label = "energy: ["
			if(len(erange) > 1):
				mask = (df_e['energy'] >= erange[0]) & (df_e['energy'] <= erange[1])
				label += str(erange[0])+", "+str(erange[1])+"] GeV"
			else:
				mask = df_e['energy'] >= erange[0]
				label += str(erange[0])+", inf) GeV"
			if mask is None:
				continue
			df_mask = df_e[mask]
			ytest = df_mask['ytrue'].to_numpy()
			ytest = [np.array(i) for i in ytest]
			ypred = df_mask['ypred'].to_numpy()
			ypred = [np.array(i) for i in ypred]
			#print("energy range",erange," - labels",np.unique(df_mask['ytrue'].to_numpy()))
			#if no entries in energy range, skip
			if(len(ytest) < 1):
				continue
			fpr, tpr = self.MakeROC(ytest, ypred,pos_label = pos_label)
			labels.append(label)
			fprs.append(fpr)
			tprs.append(tpr)
		extralab = "energySep"
		if fextra != "":
			extralab += "_"+fextra
		self.PlotROCs(fprs,tprs,labels,colors,extralab)


	def MakeTestPdDataframe(self, ypred):
		#add each score in ypred to xtest df as separate columns for plotting later
		cols = [f"score_{val[1]}" for val in self._test_gen.GetLabels()]
		chunk_size = 1024  # tune this for memory usage
		ypred_da = da.from_array(ypred, chunks=(chunk_size, ypred.shape[1]))
		scores_ddf = dd.from_dask_array(ypred_da, columns=cols)
		#get dask dataframe
		xtest_ddf = self._test_gen.GetDataFrame()
		#reset row indices
		xtest_reset = xtest_df.reset_index(drop=True)
		scores_reset = scores_ddf.reset_index(drop=True)
		#align partitions
		if self._xtest_ddf.npartitions != scores_ddf.npartitions:
			scores_ddf = scores_ddf.repartition(npartitions=xtest_ddf.npartitions)
		self._xtest_df = pd.concat([xtest_reset, scores_reset], axis=1,ignore_index=False) 
	
	def TestModelGenerator(self,batch_size=1,verb=1,validate_model = False, fpr_threshs = []):
		self.LoadBestModel()
		#save optimal model as .keras for frugally-deep
		ypred = self._model.predict(self._test_gen,batch_size=batch_size,verbose=verb)
		#get truth labels of testset
		ytest = self._test_gen.GetTrueLabels()	
	
		self.MakeTestPdDataframe(ypred)

		discr_threshs = []
		fpr_thresh = 0.001
	
		#do preprocessing for energy-separated roc curves
		energy_ranges = []
		nclasses = len(ypred[0])
		if nclasses == 2:
			labels = []
			classes = []
			for key in self._catnames.keys():
				labels.append(key)
				classes.append(self._catnames[key])
			pos_label = 1
			if labels == [4,6]:
				#know that OneHotEncoder handles labels in numerical order, so if 4 corresponds to isoBkg, then its corresponding OneHotEncoded idx is 0
				pos_label = 0
			discr_thresh = self.VizROC(ytest, ypred,class1name=classes[0],class2name=classes[1],pos_label = pos_label, fpr_threshs = fpr_threshs, fpr_thresh = fpr_thresh)
			discr_threshs.append(discr_thresh)
		else:  #multiclass
			#plot physics bkg vs other bkgs
			thresh_class1 = self.VizMulticlassROC(ytest, ypred,1,zoom=True, fpr_thresh = fpr_thresh)
			#plot BH vs other bkgs
			thresh_class2 = self.VizMulticlassROC(ytest, ypred,2,zoom=True, fpr_thresh = fpr_thresh)
			#plot spike vs other bkgs
			thresh_class3 = self.VizMulticlassROC(ytest, ypred,3,zoom=True, fpr_thresh = fpr_thresh)
			discr_threshs = [thresh_class1, thresh_class2, thresh_class3]
			#plot one-v-one for each class
			self.VizMulticlassROC(ytest, ypred,-1)
			self.VizMulticlassROC(ytest, ypred,-1,zoom=True)
		return discr_threshs
	
	def TestModel(self,batch_size=1,verb=1,validate_model = False, fpr_threshs = []):
		self.LoadBestModel()
		#save optimal model as .keras for frugally-deep
		ypred = self._model.predict(self._xtest,batch_size=batch_size,verbose=verb)
		#add each score in ypred to xtest df as separate columns
		cols = [f"score_{val[1]}" for val in enumerate(np.unique(self._lb.inverse_transform(self._ytest)))]
		scores_df = pd.DataFrame(ypred, columns=cols)
		#reset row indices
		self._xtest_df = self._xtest_df.reset_index(drop=True)
		scores_df = scores_df.reset_index(drop=True)
		self._xtest_df = pd.concat([self._xtest_df, scores_df], axis=1,ignore_index=False) 

		discr_threshs = []
		fpr_thresh = 0.001
	
		#do preprocessing for energy-separated roc curves
		energy_ranges = []
		nclasses = len(ypred[0])
		if nclasses == 2:
			labels = []
			classes = []
			for key in self._catnames.keys():
				labels.append(key)
				classes.append(self._catnames[key])
			pos_label = 1
			if labels == [4,6]:
				#know that OneHotEncoder handles labels in numerical order, so if 4 corresponds to isoBkg, then its corresponding OneHotEncoded idx is 0
				pos_label = 0
			discr_thresh = self.VizROC(self._ytest, ypred,class1name=classes[0],class2name=classes[1],pos_label = pos_label, fpr_threshs = fpr_threshs, fpr_thresh = fpr_thresh)
			discr_threshs.append(discr_thresh)
			#do energy breakdown
			if(self._xtest_energy is not None):
				self.TestModel_EnergySplit(self._xtest,self._xtest_energy,self._ytest_energy,ypred,pos_label,batch_size=batch_size,verb=verb)
		else:  #multiclass
			#plot physics bkg vs other bkgs
			thresh_class1 = self.VizMulticlassROC(self._ytest, ypred,1,zoom=True, fpr_thresh = fpr_thresh)
			#plot BH vs other bkgs
			thresh_class2 = self.VizMulticlassROC(self._ytest, ypred,2,zoom=True, fpr_thresh = fpr_thresh)
			#plot spike vs other bkgs
			thresh_class3 = self.VizMulticlassROC(self._ytest, ypred,3,zoom=True, fpr_thresh = fpr_thresh)
			discr_threshs = [thresh_class1, thresh_class2, thresh_class3]
			#plot one-v-one for each class
			self.VizMulticlassROC(self._ytest, ypred,-1)
			self.VizMulticlassROC(self._ytest, ypred,-1,zoom=True)
		return discr_threshs
	#external_class is the class you want to use ncat is the cat # you want to replace
	def ReplaceClass(self, external_class, ncat, cols, catnames, nsamp = -1, fextra=""):
		df2 = pd.DataFrame(data=self._scaler.inverse_transform(self._xtest),columns=cols) #remake dataframe with column names
		if(len(self._xtest_energy) > 0):
			df2['Energy'] = self._xtest_energy
		int_labels = self._lb.inverse_transform(self._ytest)[:,0]
		df2['label'] = int_labels
		#remove sig from df2 and replace with sig in df
		df2 = df2[df2['label'] != ncat]
		df = pd.concat([external_class,df2],ignore_index=True)

	
		#sample only n samples of dataset
		if(nsamp == -1):
			nlabels = []
			for l in np.unique(int_labels):
				nlabels.append(len(df[df['label'] == l]))
			nsamp = min(nlabels)
		sampled_subset_pos = df[df['label'] == ncat].sample(n=nsamp, random_state=42)
		sampled_subset_neg = pd.DataFrame([])
		for l in np.unique(int_labels):
			if l == ncat:
				continue
			sampled_subset_neg = pd.concat([sampled_subset_neg,df[df['label'] == l].sample(n=nsamp, random_state=42)])
		
		#replace all rows with label l by this sampled subset
		df_samp = pd.concat([sampled_subset_pos, sampled_subset_neg], ignore_index=True)
		if("Energy" in df.columns):
			energy_samp = df_samp["Energy"]
			df_samp = df_samp.drop("Energy",axis=1)

		# need to add bkg to sig BEFORE process data bc that does the plotting
		x, ytrue, _, energy = self.ProcessData(df_samp)
		xnorm = x
		self._scaler.transform(xnorm)
		ytrue_int = self._lb.inverse_transform(ytrue)[:,0]
			

		x_hist = np.array([np.array(i) for i in x])
		#make hists of training data
		y_hist = ytrue_int 
		histdata = pd.DataFrame(data=x_hist,columns=self._features)
		histdata['label'] = y_hist
		self.MakeHists(histdata,self._features,catnames,fextra) 
		
		return xnorm, ytrue, energy_samp


	#data is a 1D array
	def MakeInputHist(self, data, col, label, fextra = ""):
			plotname = self._path+"/"+col
			if(fextra != ""):
				plotname += "_"+fextra
			plotname += "."+self._form
			bins = np.linspace(data.min(), data.max(), 50)
			ns, bins, _ = plt.hist(data,label=label,log=True,bins=bins,histtype=u'step')
			plt.title(col)
			plt.legend()
			print("Saving "+col+" "+label+" plot to",plotname)
			plt.savefig(plotname,format=self._form)
			plt.close()		

	def MakeMultiInputHist(self, data, col, catnames, fextra = ""):
			plotname = self._path+"/"+col
			if(fextra != ""):
				plotname += "_"+fextra
			plotname += "."+self._form
		
			histmin = []
			histmax = []
			#print("catnames",catnames)
			for j, l in enumerate(catnames.keys()):
				mask = data['label'] == l
				histdata = data[mask][col]
				histmin.append(histdata.min())
				histmax.append(histdata.max())
			bins = np.linspace(min(histmin), max(histmax), 50)
			for j, l in enumerate(catnames.keys()):
				mask = data['label'] == l
				histdata = data[mask][col].to_numpy()
				ns, bins, _ = plt.hist(histdata,label=catnames[l],log=True,bins=bins,histtype=u'step',color = self._catcolors[l])
			plt.title(col)
			plt.legend()
			print("Saving "+col+" plot to",plotname)
			plt.savefig(plotname,format=self._form)
			plt.close()		
		
	
	def MakeHists(self, data, cols, catnames, fextra = ""):
		for i, f in enumerate(cols):
			#select column f with rows with label l
			self.MakeMultiInputHist(data,f,catnames,fextra)


	def TestModel_ExternalSignal(self,external_sig, cols,batch_size=1,verb=1,fpr_threshs = []):
		fextra = "testSet_SMS_as_signal"
		catnamesmap = {4 : "gogo", 6 : self._catnames[6]}
		x, ytrue, energy = self.ReplaceClass(external_sig,4,self._features,catnamesmap, -1,fextra)
		
		#done in ReplaceClass -> ProcessData
		if(len(energy) > 0):
			energy = np.array([[i] for i in energy])
		#normalize data for prediction
		x = self._scaler.transform(x)
		ypred = self._model.predict(x,batch_size=1,verbose=verb)
		if len(self._ytest[0]) == 2:
			labels = []
			classes = []
			for key in self._catnames.keys():
				labels.append(key)
				classes.append(self._catnames[key])
			if labels == [4,6]:
				#know that OneHotEncoder handles labels in numerical order, so if 4 corresponds to isoBkg, then its corresponding OneHotEncoded idx is 0
				pos_label = 0
			self.VizROC(ytrue, ypred,class1name="gogo",class2name="non iso bkg",pos_label=pos_label,fextra=fextra, fpr_threshs = fpr_threshs)

		if len(energy) > 0:
			#print("# energies",len(energy),"# x",len(x),"# ytrue",len(ytrue),"# ypred",len(ypred))
			energy = np.array([i[0] for i in energy])
			self.TestModel_EnergySplit(x,energy,ytrue,ypred,pos_label=pos_label,fextra=fextra)

		#else:  #multiclass
		#	#plot physics bkg vs other bkgs
		#	self.VizMulticlassROC(y, ypred,1,False,fextra)
		#	#plot one-v-one for each class
		#	self.VizMulticlassROC(y, ypred,-1,False,fextra)
		#	self.VizMulticlassROC(y, ypred,-1,True,fextra)
		#self.VizImportance()

	def SaveModelArch(self, fname):
		arch = self._model.to_json()
		fname += ".json"
		with open(fname,'w') as arch_file:
			arch_file.write(arch)
		print("Saved model architecture to",fname)

	def GetModel(self):
		if self._model is None:
			print("Model not built yet")
			return
	def VizLoss(self, history, fname):
		plt.figure()
		ax = plt.gca()
		plt.plot(history.history['val_loss'], label="val loss")
		plt.plot(history.history['loss'],label="train loss")
		ax.set_title("Loss during training",fontsize=16)
		ax.set_xlabel("Epoch",fontsize=14)
		ax.set_ylabel("Loss",fontsize=14)
		plt.legend()
		plt.savefig("loss/"+fname)
