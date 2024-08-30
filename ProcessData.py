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
		#put extra cuts on subcluster energy, etc.	
				
	def GetData(self):
		return self._data

#reader = CSVReader("csv/GJets_R17_v16_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_photons_defaultv3p2.csv")
#reader = CSVReader("csv/photonSkim_test_emAlpha0p500_thresh1p000_NperGeV0p100_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_output257_v16.csv")
#reader.CleanData()
