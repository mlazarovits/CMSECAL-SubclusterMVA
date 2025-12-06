import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import pyarrow as pa #for memory management
import pyarrow.parquet as pq
import os
import time

class FileReader:
	def __init__(self, printStats = True):
		self._data = pd.DataFrame([])
		self._printstats = printStats
		self._rechits = False
		self._tag = ""
		self._output_dir = "parquet_output"
		os.makedirs(self._output_dir,exist_ok=True)

	@abstractmethod
	def AddFile(self, file):
		pass

	def CleanData(self):
		print("Cleaning data",len(self._data),"subclusters initially")

		#remove any "unmatched" labels
		self._data = self._data[self._data['label'] != -1]
		if(self._printstats):
		    print("after unmatched removal")
		    self.PrintStats()

		self._data = self._data[self._data['label'] != -999]
		if(self._printstats):
		    print("after invalid reconstruction removal")
		    self.PrintStats()
		
		#put extra cuts on subcluster energy, etc.
		if "Energy" in self._data.columns:
			self.ApplyColCut("Energy",30) 
		
		#drop nan rows
		for col in self._data.columns:
		    if(self._data[col].isna().any()):
		        print("column",col,"has nans in rows")
		self._data.dropna(how="any")
		if(self._printstats):
		    print('after dropna')
		    self.PrintStats()

	def PrintStats(self):
		phys = len(self._data[self._data["label"] == 1])
		BH = len(self._data[self._data["label"] == 2])
		spike = len(self._data[self._data["label"] == 3])
		tot = phys + BH + spike
		if(BH > 0):
		        print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
		isobkg = len(self._data[self._data["label"] == 4])
		nonisobkg = len(self._data[self._data["label"] == 6])
		tot = len(self._data)
		if(isobkg > 0):
			print(" ",tot, ("subclusters, isobkg: "+str(isobkg)+" {:.2f}%, nonisobkg: "+str(nonisobkg)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
    
	#keep rows with values in col > val
	def ApplyColCut(self, col, val):
		self._data = self._data[self._data[col] > val]
	
	
	def SelectClass(self,nclass,samp):
		mask = (self._data["label"] == nclass) | (self._data["sample"] != samp)
		self._data = self._data[mask]
		'''
		#drop rows from all samps that are not nclass
		self._data = self._data[~((self._data["sample"].isin(samps)) & (self._data["label"] != nclass))]
		##drop rows for !samp that are nclass
		self._data = self._data[~((self._data["label"] == nclass) & (~self._data["sample"].isin(samps)))]
		'''	
		if(self._printstats):
		    print("after setting class",nclass,"to be only from",samp)
		    self.PrintStats()
	
	#only allow nsamp of samples from nclass
	def CapClass(self,nclass,nsamp):
		sampled_subset = self._data[self._data['label'] == nclass].sample(n=nsamp, random_state=42)	
		#replace all rows with label l by this sampled subset
		self._data = pd.concat([self._data[self._data['label'] != nclass], sampled_subset], ignore_index=True)
	
	#cut off samples depending on feature < val
	def CapFeature(self,nclass,feature,val):
		classcond = self._data["label"] == nclass
		featcond = self._data[feature] >= val
		#remove rows that satisfy both of the above conditions
		self._data = self._data[~(classcond & featcond)]
	
	#reweigh target_class to bench_class
	#apply weights to target class
	def ReweightClasses(self,bench_class,target_class,feature):
		feat_bench = self._data.loc[self._data['label'] == bench_class, feature]
		feat_target = self._data.loc[self._data['label'] == target_class, feature]
	
		bins = np.linspace(self._data[feature].min(), self._data[feature].max(), 50)
	
		#reweigh histograms
		hist_bench, _ = np.histogram(feat_bench,bins=bins,density=True)
		hist_target, _ = np.histogram(feat_target,bins=bins,density=True)
	
		#compute ratio for reweighting (w = bench / target s.t. w*target = bench)
		#ie reweight target to match bench
	
		ratio = np.divide(hist_bench, hist_target, out=np.zeros_like(hist_target), where=hist_target>0)
	
		#assign weight to each target sample
		bin_indices = np.digitize(feat_target, bins) - 1
		weights = [ratio[x] if x < len(ratio) else 0 for x in bin_indices]
		#weights = ratio[bin_indices]
	
		#add to dataframe
		self._data.loc[self._data["label"] == target_class, "weight"] = weights
		self._data.loc[self._data["label"] == bench_class, "weight"] = 1.0
	
	
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
	
	#creates new columns that are ratios of given cols and denom column
	def DivideCols(self, cols, denom):
		newnames = []
		for col in cols:
			colname = col+'Ov'+denom
			newnames.append(colname)
			self._data[colname] = self._data[col] / self._data[denom]
			#print("col",self._data[self._data.isna().any(axis=1)][colname])
		return newnames
	
	def MakeSigmas(self, cols):
		for col in cols:
			if "Var" not in col:
				continue
			newcolname = col[:col.find("Var")]+"Sig"
			self._data[newcolname] = np.sqrt(self._data[col])
	
	def SetFeatureToVal(self, feature, val):
		self._data[feature] = val
	
	def SetFeatureFromValToVal(self, feature, oldval, newval):
		self._data.loc[self._data[feature] == oldval, feature] = newval
	
	def RemoveEntries(self, feature, val):
		mask = self._data[feature] == val
		self._data = self._data[~mask]
	
	
	def BarrelOnly(self, etabranch = ""):
		if etabranch == "":
			etabranch = "EtaCenter"
		mask = ((self._data[etabranch] > -1.5) & (self._data[etabranch] < 1.5))
		self._data = self._data[mask]
		if(self._printstats):
			print('after selecting for barrel')
			self.PrintStats()
	
	
	def EndcapOnly(self,etabranch = ""):
		if etabranch == "":
			etabranch = "EtaCenter"
		mask = ((self._data[etabranch] > -1.5) & (self._data[etabranch] < 1.5))
		self._data = self._data[~mask]
		if(self._printstats):
			print('after selecting for endcap')
			self.PrintStats()


class TTreeReader(FileReader):
	def __init__(self, tag, printStats = False):
		super().__init__(printStats)
		self._tag = tag
 
	def ProcessCNNBranches(self, file, sample, step_size=10000):
		branches = [
			f"SC_rh_iEta_{self._tag}",
			f"SC_rh_iPhi_{self._tag}",
			f"SC_rh_Energy_{self._tag}",
			f"SC_rh_Weight_{self._tag}",
			f"SC_trueLabel_{self._tag}",
			f"SC_EtaCenter_{self._tag}"
		]
		print("Branches",branches)	
		data_accum = []
		nchunk = 0
		total_time = 0

		nentries = uproot.open(file)["tree"].num_entries
		if(isinstance(step_size,int)):
			print("File",file,"has",nentries,"entries and therefore",(nentries + step_size - 1) // step_size,"chunks with step size",step_size)
		else:
			print("Chunking file",file,"in",step_size,"chunks")
		for chunk in uproot.iterate(file + ":tree", branches, step_size=step_size, library="ak",
				num_workers = 8, #multithreading options
				decompression_executor=ThreadPoolExecutor(max_workers=8),
				interpretation_executor=ThreadPoolExecutor(max_workers=8)
			):
			t1 = time.perf_counter()
			if nchunk > 300 and "EGamma" in sample:
				break
			print(f"Processing chunk #{nchunk}",end="\r",flush=True)
			
			sc_counts = ak.num(chunk[f"SC_trueLabel_{self._tag}"])
			#flatten sc level branches
			sc_eta = ak.flatten(chunk[f"SC_EtaCenter_{self._tag}"],axis=1)
			sc_label = ak.flatten(chunk[f"SC_trueLabel_{self._tag}"],axis=1)
			#flatten rh level branches
			rechit_branches = {
				f"SC_rh_iEta_{self._tag}": chunk[f"SC_rh_iEta_{self._tag}"],
				f"SC_rh_iPhi_{self._tag}": chunk[f"SC_rh_iPhi_{self._tag}"],
				f"SC_rh_Energy_{self._tag}": chunk[f"SC_rh_Energy_{self._tag}"],
				f"SC_rh_Weight_{self._tag}": chunk[f"SC_rh_Weight_{self._tag}"]
			}

			#make index branches
			event_idx = np.repeat(np.arange(len(chunk)), sc_counts)
			sc_idx = ak.flatten(ak.local_index(chunk[f"SC_trueLabel_{self._tag}"]))

			data = {
				"event_idx" : event_idx,
				"sc_idx" : sc_idx.to_numpy(), 
				f"SC_EtaCenter_{self._tag}" : sc_eta.to_numpy(),
				f"SC_trueLabel_{self._tag}" : sc_label.to_numpy(),
			}
			for rh in rechit_branches:
				# flatten outer axis only (keep inner list of rechits per SC)
				data[rh] = ak.to_list(ak.flatten(chunk[rh], axis=1))

			data["sample"] = sample
			#convert to arrow table and write to parquet to avoid large RAM usage all at once
			table = pa.Table.from_pandas(pd.DataFrame(data))
			pq.write_table(table, os.path.join(self._output_dir, f"chunk_{nchunk:05d}_sample_{sample}.parquet"))

			#data_accum.append(df)
			nchunk += 1
			t2 = time.perf_counter()
			total_time += (t2 - t1)
			print(f"Chunk #{nchunk} processed took",(t2-t1),"seconds",end="\r",flush = True)
		
		# Concatenate all chunks into single DataFrame
		#data_accum.append(self._data)
		#self._data = pd.concat(data_accum, ignore_index=True)
		print("Done processing file",file,"took",total_time,"seconds total with",total_time / nchunk,"seconds on average per chunk\n")
	
	def AddFileCNN(self, file, sample, step_size=10000):
		self._rechits = True
		self.ProcessCNNBranches(file,sample,step_size)
		

	def ReadDataFromParquetTable(self):
		dataset = pq.ParquetDataset(self._output_dir)
		table = dataset.read()
		return table.to_pandas()	

	def CleanData(self):
		print("Reading from parquet table")
		t1 = time.perf_counter()
		self._data = self.ReadDataFromParquetTable()
		t2 = time.perf_counter()
		print("took",(t2-t1),"to read data from parquet table")
		self._data.rename(columns={f"SC_trueLabel_{self._tag}" : "label", f"SC_EtaCenter_{self._tag}" : "SC_EtaCenter"}, inplace=True)
		print("All chunks processed, total rows:", len(self._data),"beam halo SCs:",len(self._data[self._data["label"] == 2]),"spike SCs:",len(self._data[self._data["label"] == 3]),"physics SCs:",len(self._data[self._data["label"] == 1]))
		super().CleanData()

#add multiple files
class CSVReader(FileReader):
	def __init__(self, file, printStats = False):
		#data is a list of feature, label pairs or tuples
		super().__init__(printStats)
		self._file = file
		self._header = np.array([])
		self._data = pd.read_csv(self._file)

		#dictionary for integer labels to strings for plotting
		self._labelsDict = {0 : "sig", 1 : "physics", 2 : "BH", 3 : "spike", 4 : "isoBkg", 6 : "nonIsoBkg"}
	
	def __init__(self, printStats = False):
		#data is a list of feature, label pairs or tuples
		super().__init__(printStats)
		self._header = np.array([])
		self._data = pd.DataFrame()
		#dictionary for integer labels to strings for plotting
		self._labelsDict = {0 : "sig", 1 : "physics", 2 : "BH", 3 : "spike", 4 : "isoBkg", 6 : "nonIsoBkg"}

	def AddFile(self,file):
		data = pd.read_csv(file)
		#add sample col if doesnt exist
		if "sample" not in data.columns:
			sample = file[:file.find("_R")]
			sample = sample[sample.rfind("_")+1:]
			data["sample"] = sample
			self._data = pd.concat([self._data,data],ignore_index=True)

	def AddLargeFile(self,file,nsamp=1000,chunksize=1e5):
		if nsamp > chunksize:
			print("nsamp",nsamp,"cannot be larger than chunksize",chunksize)
			return
		largedata = []
		if not self._data.empty:
			largedata = largedata.append(self._data)
		#total # of samples will be floor(nrows/chunksize)*nsamp
		for nchunk, chunk in enumerate(pd.read_csv(file,chunksize=chunksize)):
			if(len(largedata) > 10 and nsamp != -1):
				break
			print("processing chunk #",nchunk)
			if(nsamp != -1):
				chunk = chunk.sample(n=nsamp,random_state=111)
			largedata.append(chunk)
		self._data = pd.concat(largedata,ignore_index=True)
		print("finishing concating dfs - have total of",len(self._data),"rows")
		

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

