import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

#add multiple files
class CSVReader:
	def __init__(self, file):
		#data is a list of feature, label pairs or tuples
		self._data = np.array([])
		self._file = file
		self._header = np.array([])
		self._data = pd.read_csv(self._file)
	
	def __init__(self):
		#data is a list of feature, label pairs or tuples
		self._data = np.array([])
		self._header = np.array([])
		self._data = pd.DataFrame()

	def AddFile(self,file):
		data = pd.read_csv(file)
		self._data = pd.concat([self._data,data],ignore_index=True)

	def PrintStats(self):
		sig = len(self._data[self._data["label"] == 0])
		nom = len(self._data[self._data["label"] == 1])
		gmsb = len(self._data[self._data["sample"].str.contains("GMSB") == True])
		gjets = len(self._data[self._data["sample"].str.contains("GJets") == True])
		qcd = len(self._data[self._data["sample"].str.contains("QCD") == True])
		d = len(self._data[self._data["sample"] == "METPD"]) + len(self._data[self._data["sample"] == "EGamma"]) 
		tot = len(self._data)
		print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, GMSB "+str(gmsb)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	
		
	#data cleaning, cuts, etc.
	#unmatched = -1
	#signal = 0
	#!signal = 1 
	#BH = 2
	#spike = 3
	def CleanData(self, printStats = False):
		print("Cleaning data",len(self._data),"subclusters initially")
		#remove any "unmatched" labels
		self._data = self._data[self._data['label'] != -1]
		if(printStats):
			print("after unmatched removal")
			self.PrintStats()
	
		#remove not-signal-matched photons in GMSB sample (this is not the "bkg" we want to target)
		rowbool = ((self._data["sample"].str.contains("GMSB") == True) & (self._data["label"] == 1)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		if(printStats):
			print("after GMSB bkg removal")
			self.PrintStats()

		#remove phys entries from data samples - want these entries to come from MC
		rowbool = ((self._data["sample"].str.contains("METPD") == True) & (self._data["label"] == 1)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		rowbool = ((self._data["sample"].str.contains("JetHT") == True) & (self._data["label"] == 1)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		rowbool = ((self._data["sample"].str.contains("EGamma") == True) & (self._data["label"] == 1)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		if(printStats):
			print("after data bkg removal")
			self.PrintStats()
		
		#put extra cuts on subcluster energy, etc.	
		self._data.dropna()
		if(printStats):
			print('after dropna')
			self.PrintStats()


	def TransformLabels(self):
		lb = LabelBinarizer()
		labels = self._data["label"]
		newlabels = lb.fit_transform(labels)
		self._data = self._data.assign(label = pd.Series(list(newlabels)).values)
		print(self._data["label"])
	

	def GetData(self):
		return self._data

