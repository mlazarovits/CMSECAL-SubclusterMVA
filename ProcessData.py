import numpy as np

class CSVReader:
	def __init__(self, file):
		#data is a list of feature, label pairs or tuples
		self._data = np.array([])
		self._file = file

	def ExtractData(self):
		f = open(self._file)
		header = f.readline().split(", ")
		print(header)
		basedata = np.genfromtxt(self._file, delimiter=",",skip_header=1)
		dataset = [list(tup) for tup in basedata]
		#separate labels from features here and store as ([features], label) in self._data
		dataset = np.array(dataset)	
		print(dataset.shape)
		print(dataset[0])	
		
	#data cleaning, cuts, etc.
	#def CleanData(self):
			
					
reader = CSVReader("csv/GJets_R17_v16_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_photons_defaultv3p2.csv")
reader.ExtractData()
