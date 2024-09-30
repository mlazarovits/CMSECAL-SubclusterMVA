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
		self._data = pd.concat([self._data,data])

		
	#data cleaning, cuts, etc.
	def CleanData(self):
		print("Cleaning data")
		#remove any "unmatched" labels
		self._data = self._data[self._data['label'] != -1]
		#remove not-signal-matched photons in GMSB sample (this is not the "bkg" we want to target)
		rowbool = (self._data["sample"].str.contains("GMSB")) & (self._data["label"] == 1) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		#put extra cuts on subcluster energy, etc.	
				
	def GetData(self):
		return self._data

