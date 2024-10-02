import numpy as np
import pandas as pd

#add multiple files
class CSVReader:
	def __init__(self, file):
		#data is a list of feature, label pairs or tuples
		self._data = np.array([])
		self._file = file
		self._header = np.array([])
		self._data = pd.read_csv(self._file)

	def AddFile(self,file):
		data = pd.read_csv(file)
		self._data = pd.concat([self._data,data],ignore_index=True)
	
		
	#data cleaning, cuts, etc.
	def CleanData(self):
		print("Cleaning data",len(self._data))
		#remove any "unmatched" labels
		self._data = self._data[self._data['label'] != -1]
		print("after -1 removal",len(self._data))
		sig = len(self._data[self._data["label"] == 0])
		nom = len(self._data[self._data["label"] == 1])
		gmsb = len(self._data[self._data["sample"].str.contains("GMSB") == True])
		gjets = len(self._data[self._data["sample"].str.contains("GJets") == True])
		print("sig",sig,"nom",nom,"gmsb",gmsb,"gjets",gjets)
		#remove not-signal-matched photons in GMSB sample (this is not the "bkg" we want to target)
		rowbool = ((self._data["sample"].str.contains("GMSB") == True) & (self._data["label"] == 1)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		print("after GMSB bkg removal",len(self._data))
		sig = len(self._data[self._data["label"] == 0])
		nom = len(self._data[self._data["label"] == 1])
		gmsb = len(self._data[self._data["sample"].str.contains("GMSB") == True])
		gjets = len(self._data[self._data["sample"].str.contains("GJets") == True])
		print("sig",sig,"nom",nom,"gmsb",gmsb,"gjets",gjets)
		#put extra cuts on subcluster energy, etc.	
		#remove nans - TODO: see why this is happening with minor_length
		self._data.dropna()
		print('after dropna')
		sig = len(self._data[self._data["label"] == 0])
		nom = len(self._data[self._data["label"] == 1])
		gmsb = len(self._data[self._data["sample"].str.contains("GMSB") == True])
		gjets = len(self._data[self._data["sample"].str.contains("GJets") == True])
		print("sig",sig,"nom",nom,"gmsb",gmsb,"gjets",gjets)
				
	def GetData(self):
		return self._data

