import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#add multiple files
class CSVReader:
	def __init__(self, file, printStats = False):
		#data is a list of feature, label pairs or tuples
		self._data = np.array([])
		self._file = file
		self._header = np.array([])
		self._data = pd.read_csv(self._file)
		#dictionary for integer labels to strings for plotting
		self._labelsDict = {0 : "sig", 1 : "physics", 2 : "BH", 3 : "spike"}
		self._printstats = printStats
	
	def __init__(self, printStats = False):
		#data is a list of feature, label pairs or tuples
		self._data = np.array([])
		self._header = np.array([])
		self._data = pd.DataFrame()
		#dictionary for integer labels to strings for plotting
		self._labelsDict = {0 : "sig", 1 : "physics", 2 : "BH", 3 : "spike"}
		self._printstats = printStats

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
		phys = len(self._data[self._data["label"] == 1])
		BH = len(self._data[self._data["label"] == 2])
		spike = len(self._data[self._data["label"] == 3])
		print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
		print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, GMSB "+str(gmsb)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	
		
	#data cleaning, cuts, etc.
	#unmatched = -1
	#signal = 0
	#!signal = 1 
	#BH = 2
	#spike = 3
	def CleanData(self):
		print("Cleaning data",len(self._data),"subclusters initially")
		#remove any "unmatched" labels
		self._data = self._data[self._data['label'] != -1]
		if(self._printstats):
			print("after unmatched removal")
			self.PrintStats()
	
		#remove not-signal-matched photons in GMSB sample (this is not the "bkg" we want to target)
		rowbool = ((self._data["sample"].str.contains("GMSB") == True) & (self._data["label"] == 1)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		if(self._printstats):
			print("after GMSB bkg removal")
			self.PrintStats()

		#put extra cuts on subcluster energy, etc.	
		self._data.dropna()
		if(self._printstats):
			print('after dropna')
			self.PrintStats()

	#only use SCs with 1 subcluster
	#can remove if CSVs are updated accordingly
	def LeadingOnly(self):
		#if entry in (evt, object) appears multiple times, skip all rows with (evt, obj)
		maxevtnum = max(self._data["event"])+1
		for e in range(maxevtnum):
			if len(self._data[self._data["event"] == e]) < 1:
				continue
			maxobjnum = max(self._data[self._data["event"] == e]["object"])+1
			for o in range(maxobjnum):
				rowbool = ((self._data["event"] == e) & (self._data["object"] == o))
				nEntries = len(self._data[rowbool])
				if nEntries > 1:
					#print("event",e,"obj",o,"has",nEntries,"subcls with max energy",maxenergy)
					#drop all rows with this event and object
					self._data = self._data.drop(self._data[rowbool].index)
		if(self._printstats):
			print('after dropping SCs with multiple subclusters')
			self.PrintStats()

	#remove subleading subclusters
	def RemoveSubleading(self):
		#save + return dataframe with rows of all subleading subclusters
		sublead_data = self._data[self._data['subcl'] != 0]
		self._data = self._data[self._data['subcl'] == 0]
		if(self._printstats):
			print('after removing subleading subclusters')
			self.PrintStats()
		return sublead_data


	def SelectClass(self,nclass,samp):
		#drop rows from samp that are not nclass
		rowbool = ((self._data["sample"].str.contains(samp) == True) & (self._data["label"] != nclass)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		#drop rows for !samp that are nclass
		rowbool = ((self._data["sample"].str.contains(samp) == False) & (self._data["label"] == nclass)) 
		self._data = self._data.drop(self._data[rowbool].index)	 
		if(self._printstats):
			print("after setting class",nclass,"to be only from",samp)
			self.PrintStats()

	def BalanceClasses(self, labels):
		sizes = {}
		for l in labels:
			sizes[len(self._data[self._data["label"] == l])] = l
		nsamp = min(sizes.keys())
		lab = sizes[nsamp] #label of min sample
		
		data_samples = []	
		for l in sizes.values():
			if l == lab:
				continue	
			data_samples.append(self._data.query('label == '+str(l)).sample(n=nsamp,random_state=111))
		data_samples.append(self._data.query('label == '+str(lab)))
		self._data = pd.concat(data_samples)
		if(self._printstats):
			print('after balancing classes')
			self.PrintStats()

	def GetData(self):
		return self._data

