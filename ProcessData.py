import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import pyarrow as pa #for memory management
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import dask.dataframe as dd
import os
import time

class FileReader:
	def __init__(self, obj, printStats = True):
		self._data = pd.DataFrame([])
		self._printstats = printStats
		self._tag = ""
		self._output_parquet_data = "parquet_output"
		os.makedirs(self._output_parquet_data,exist_ok=True)
		self._obj = obj

	def SetParquetOutputDir(self, pdir):
		if not os.path.exists(pdir):
			os.mkdir(pdir)
		self._output_parquet_data = pdir

	def SetPrintStats(self, p):
		self._printstats = p

	@abstractmethod
	def AddFile(self, file):
		pass
	'''
	def CleanData(self, indata = None):
		if(indata is None):
			indata = self._data
		if(self._printstats):
			print("Cleaning data",len(indata),"subclusters initially")

		#remove any "unmatched" labels
		indata = indata[indata['label'] != -1]
		if(self._printstats):
		    print("after unmatched removal")
		    self.PrintStats(indata)

		indata = indata[indata['label'] != -999]
		if(self._printstats):
		    print("after invalid reconstruction removal")
		    self.PrintStats(indata)
		
		#put extra cuts on subcluster energy, etc.
		if "Energy" in indata.columns:
			self.ApplyColCut("Energy",30, indata) 
		
		#drop nan rows
		for col in indata.columns:
		    if(indata[col].isna().any()):
		        print("column",col,"has nans in rows")
		indata.dropna(how="any")
		if(self._printstats):
		    print('after dropna')
		    self.PrintStats(indata)
	'''

	def CleanDataDask(self, indata=None):
		"""
		Cleans the input DataFrame or Dask DataFrame:
		- removes invalid/unmatched labels
		- applies column cuts (e.g., Energy)
		- drops rows with NaNs
		Returns a cleaned Dask DataFrame (lazy until compute()).
		"""
	
		#  Use provided input or default dataset
		if indata is None:
		         indata = self._data  # could be pandas or Dask DataFrame
	
		#  Print initial stats (compute row count lazily)
		if self._printstats:
			nrows = indata.shape[0].compute()  # works for Dask
			print("Cleaning data", nrows, self._obj+"s","initially")
			self.PrintStatsDask(indata)
				
		#  Remove invalid labels (lazy, memory-efficient)
		indata = indata.query("label != -1 and label != -999")
		if self._printstats:
		       print("After unmatched/invalid label removal:", indata.shape[0].compute())
		       self.PrintStatsDask(indata)

		#  Apply column cuts (e.g., Energy)
		if "Energy" in indata.columns:
			print("Applying energy cut")
			# Ensure ApplyColCut works with Dask: avoid .values
			indata = self.ApplyColCut("Energy", 30, indata)
			if self._prinstats:
				print("after energy cut > 30")
				self.PrintStatsDask(indata)
	
		#  Drop rows with any NaNs
		# Lazy operation; assign back
		indata = indata.dropna(how="any")
		# check for NaNs per column (compute only scalars) - takes too long
		#for col in indata.columns:
		#	has_nan = indata[col].isna().any().compute()  # lazy scalar
		#	if has_nan:
		#		print("Warning: column", col, "still has NaNs")
	
		if self._printstats:
			print("After dropna:", len(indata))
			self.PrintStatsDask(indata)
	
		return indata

	def PrintStatsDask(self, data = None):
		if data is None:
			data = self._data
		phys = (data["label"] == 1).sum().compute()
		BH = (data["label"] == 2).sum().compute()
		spike = (data["label"] == 3).sum().compute()
		tot = phys + BH + spike
		if(BH > 0):
		        print(" ",tot, (self._obj+"s, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
		isobkg = (data["label"] == 4).sum().compute()
		nonisobkg = (data["label"] == 6).sum().compute()
		tot = len(data)
		if(isobkg > 0):
			print(" ",tot, (self._obj+"s, isobkg: "+str(isobkg)+" {:.2f}%, nonisobkg: "+str(nonisobkg)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))

	'''
	def PrintStats(self, data = None):
		if data is None:
			data = self._data
		phys = len(data[data["label"] == 1])
		BH = len(data[data["label"] == 2])
		spike = len(data[data["label"] == 3])
		tot = phys + BH + spike
		if(BH > 0):
		        print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
		isobkg = len(data[data["label"] == 4])
		nonisobkg = len(data[data["label"] == 6])
		tot = len(data)
		if(isobkg > 0):
			print(" ",tot, ("subclusters, isobkg: "+str(isobkg)+" {:.2f}%, nonisobkg: "+str(nonisobkg)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
   	''' 
	#keep rows with values in col > val
	def ApplyColCut(self, col, val, indata = None):
		if indata is None:
			indata = self._data
		indata = indata[indata[col] > val]
	
	#keep rows with values in col > val
	def ApplyColCutDask(self, col, val, indata = None):
		if indata is None:
			indata = self._data
		indatafiltered = indata[indata[col] > val]
		return indatafiltered
	
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
		denomname = denom
		if self._obj not in denom:
			denom = self._obj+"_"+denom
		newnames = []
		for col in cols:
			colname = col+'Ov'+denomname
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
	def __init__(self, obj, tag, printStats = False):
		super().__init__(obj, printStats)
		self._tag = tag
		self._output_parquet_data = self._output_parquet_data+f"/{self._tag}_{self._obj}s"
 
	def ProcessCNNBranches(self, file, sample, step_size=10000, recreate_files = False):
		branches = [
			f"SC_rh_iEta_{self._tag}",
			f"SC_rh_iPhi_{self._tag}",
			f"SC_rh_Energy_{self._tag}",
			f"SC_trueLabel_{self._tag}",
			f"SC_EtaCenter_{self._tag}",
			f"SC_seedTime_CMS"
		]
		print("Branches",branches)
		data_accum = []
		nchunk = 0
		total_time = 0

		print("Processing sample",sample,"from file",file,"with step size",step_size)
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
			parquet_fname = f"chunk_{nchunk:05d}_sample_{sample}_type_{self._tag}.parquet"
			if(os.path.exists( os.path.join(self._output_parquet_data, parquet_fname) )) and not recreate_files:
				#print(os.path.join(self._output_parquet_dataset, parquet_fname),"exists. Please provide a unique sample name for",file)
				#print(os.path.join(self._output_parquet_dataset, parquet_fname),"exists. Skipping.",end="\r",flush=True)
				return
			t1 = time.perf_counter()
			if nchunk > 800 and "EGamma" in sample:
				break
			print(f"Processing chunk #{nchunk}",end="\r",flush=True)
			
			sc_counts = ak.num(chunk[f"SC_trueLabel_{self._tag}"])
			#write to parquet table directly
			table = pa.table({
				"event_idx": np.repeat(np.arange(len(chunk)), sc_counts),
				"sc_idx": ak.to_numpy(ak.flatten(ak.local_index(chunk[f"SC_trueLabel_{self._tag}"]))),
				f"SC_EtaCenter_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"SC_EtaCenter_{self._tag}"],axis=1)),
				f"SC_seedTime_CMS" : ak.to_numpy(ak.flatten(chunk[f"SC_seedTime_CMS"],axis=1)), #ak.to_numpy must be flat arrays
				f"SC_trueLabel_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"SC_trueLabel_{self._tag}"],axis=1)),
				f"SC_rh_iEta_{self._tag}" : ak.to_list(ak.flatten(chunk[f"SC_rh_iEta_{self._tag}"], axis=1)), #ak.to_list can be jagged arrays
				f"SC_rh_iPhi_{self._tag}" : ak.to_list(ak.flatten(chunk[f"SC_rh_iPhi_{self._tag}"], axis=1)),
				f"SC_rh_Energy_{self._tag}" : ak.to_list(ak.flatten(chunk[f"SC_rh_Energy_{self._tag}"], axis=1)),
				"sample": pa.array([sample] * sum(sc_counts))
			})	

			#table = pa.Table.from_pandas(pd.DataFrame(data))
			pq.write_table(table, os.path.join(self._output_parquet_data, parquet_fname))

			#data_accum.append(df)
			nchunk += 1
			t2 = time.perf_counter()
			total_time += (t2 - t1)
			print(f"Chunk #{nchunk} processed took",(t2-t1),"seconds",end="\r",flush = True)
		
		# Concatenate all chunks into single DataFrame
		#data_accum.append(self._data)
		#self._data = pd.concat(data_accum, ignore_index=True)
		print("Done processing file",file,"took",total_time,"seconds total with",total_time / nchunk,"seconds on average per chunk\n\n")
	
	def AddFileCNN(self, file, sample, step_size=10000):
		self.ProcessCNNBranches(file,sample,step_size)
		
	def ProcessDNNBranches(self, file, sample, step_size=10000, recreate_files = False):
		branches = [
			f"Photon_EtaVar_{self._tag}",
			f"Photon_PhiVar_{self._tag}",
			f"Photon_EtaPhiCov_{self._tag}",
			f"Photon_majorLength_{self._tag}",
			f"Photon_minorLength_{self._tag}",
			f"Photon_hcalTowerSumEtConeDR04_{self._tag}",
			f"Photon_trkSumPtSolidConeDR04_{self._tag}",
			f"Photon_trkSumPtHollowConeDR04_{self._tag}",
			f"Photon_hadTowOverEM_{self._tag}",
			f"Photon_ecalRHSumEtConeDR04_{self._tag}"
			f"Photon_Pt_{self._tag}",
			f"Photon_EtaCenter_{self._tag}",
			f"Photon_trueLabel_{self._tag}"
			
		]
		print("Branches",branches)
		data_accum = []
		nchunk = 0
		total_time = 0

		print("Processing sample",sample,"from file",file,"with step size",step_size)
		nentries = uproot.open(file)["tree"].num_entries
		if(isinstance(step_size,int)):
			print("File",file,"has",nentries,"entries and therefore",(nentries + step_size - 1) // step_size,"chunks with step size",step_size)
		else:
			print("Chunking file",file,"in",step_size,"chunks")
		self._output_parquet_data = self._output_parquet_data+f"/{self._tag}_Photons"
		

		for chunk in uproot.iterate(file + ":tree", branches, step_size=step_size, library="ak",
				num_workers = 8, #multithreading options
				decompression_executor=ThreadPoolExecutor(max_workers=8),
				interpretation_executor=ThreadPoolExecutor(max_workers=8)
			):
			parquet_fname = f"chunk_{nchunk:05d}_sample_{sample}_type_{self._tag}.parquet"
			if(os.path.exists( os.path.join(self._output_parquet_data, parquet_fname) )) and not recreate_files:
				#print(os.path.join(self._output_parquet_data, parquet_fname),"exists. Please provide a unique sample name for",file)
				#print(os.path.join(self._output_parquet_data, parquet_fname),"exists. Skipping.",end="\r",flush=True)
				return
			t1 = time.perf_counter()
			print(f"Processing chunk #{nchunk}",end="\r",flush=True)
			
			pho_counts = ak.num(chunk[f"Photon_trueLabel_{self._tag}"])
			#write to parquet table directly
			table = pa.table({
				"event_idx": np.repeat(np.arange(len(chunk)), pho_counts),
				"pho_idx": ak.to_numpy(ak.flatte(ak.local_index(chunk[f"Photon_trueLabel_{self._tag}"]))),
				f"Photon_trueLabel_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"Photon_trueLabel_{self._tag}"],axis=1)),
				f"Photon_EtaVar_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"Photon_EtaVar_{self._tag}"],axis=1)),
				f"Photon_PhiVar_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"Photon_PhiVar_{self._tag}"],axis=1)),
				f"Photon_EtaPhiCov_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_EtaPhiCov_{self._tag}"],axis=1)),
				f"Photon_majorLength_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_majorLength_{self._tag}"],axis=1)),
				f"Photon_minorLength_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_minorLength_{self._tag}"],axis=1)),
				f"Photon_hcalTowerSumEtConeDR04_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_hcalTowerSumEtConeDR04_{self._tag}"],axis=1)),
				f"Photon_trkSumPtSolidConeDR04_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_trkSumPtSolidConeDR04_{self._tag}"],axis=1)),
				f"Photon_trkSumPtHollowConeDR04_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_trkSumPtHollowConeDR04_{self._tag}"],axis=1)),
				f"Photon_hadTowOverEM_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_hadTowOverEM_{self._tag}"],axis=1)),
				f"Photon_ecalRHSumEtConeDR04_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_ecalRHSumEtConeDR04_{self._tag}"],axis=1)),
				f"Photon_Pt_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_Pt_{self._tag}"],axis=1)),
				f"Photon_EtaCenter_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_EtaCenter_{self._tag}"],axis=1)),
				"sample": pa.array([sample] * sum(pho_counts))
			})	

			pq.write_table(table, os.path.join(self._output_parquet_data, parquet_fname))

			#data_accum.append(df)
			nchunk += 1
			t2 = time.perf_counter()
			total_time += (t2 - t1)
			print(f"Chunk #{nchunk} processed took",(t2-t1),"seconds",end="\r",flush = True)
		
		print("Done processing file",file,"took",total_time,"seconds total with",total_time / nchunk,"seconds on average per chunk\n\n")
	
	def AddFileDNN(self, file, sample, step_size=10000):
		self.ProcessDNNBranches(file,sample,step_size)

	def ReadDataFromParquetTable(self):
		dataset = pq.ParquetDataset(self._output_parquet_data)
		table = dataset.read()
		return table.to_pandas()	


	def CleanDataDask(self, debug = False):
		print("Reading from parquet files at",self._output_parquet_data,"into Dask df")
		t1 = time.perf_counter()
		parquet_path = self._output_parquet_data+"/*.parquet"
		if debug:
			parquet_path = self._output_parquet_data+"/chunk_00000_sample_*.parquet"
		

		ddf = dd.read_parquet(parquet_path)
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to read data from parquet table, total # rows",ddf.shape[0].compute())
		#rename cols
		new_cols = ["label" if col == f"{self._obj}_trueLabel_{self._tag}" else col for col in ddf.columns]
		if self._obj == "SC":
			new_cols = [f"{self._obj}_EtaCenter" if col == f"{self._obj}_EtaCenter_{self._tag}" else col for col in new_cols]
			new_cols = [f"{self._obj}_seedTime" if col ==  f"{self._obj}_seedTime_CMS" else col for col in new_cols]
		ddf = ddf.rename(columns=dict(zip(ddf.columns, new_cols))) #inplace not supported for dask dfs!! (lazy execution remember??)
		self.PrintStatsDask(ddf)	

		print("cleaning dask data")
		t1 = time.perf_counter()
		cleaned_ddf = super().CleanDataDask(ddf)
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to clean dask data")
		self._data = cleaned_ddf.compute()
	
	'''	
	def CleanData(self):
		print("Reading from parquet table into Dask df")
		t1 = time.perf_counter()
		dataset = ds.dataset(self._output_parquet_data, format="parquet")
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to read data from parquet table")
		dfs = []
		batchidx = 0
		t1 = time.perf_counter()
		print("batch cleaning df")
		for batch in dataset.to_batches(batch_size=1000):
			print("Cleaning batch #",batchidx,end="\r",flush=True)
			df = batch.to_pandas()
			df.rename(columns={f"SC_trueLabel_{self._tag}" : "label", f"SC_EtaCenter_{self._tag}" : "SC_EtaCenter", f"SC_seedTime_CMS" : "SC_seedTime"}, inplace=True)
			super().CleanData(df)
			#may need to do all processing in this loop and then read in samples when done
			dfs.append(df)
			batchidx += 1
		self._data = pd.concat(dfs, ignore_index=True)	
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to clean Dask df batches")
		print("All chunks processed, total rows:", len(self._data),"beam halo SCs:",len(self._data[self._data["label"] == 2]),"spike SCs:",len(self._data[self._data["label"] == 3]),"physics SCs:",len(self._data[self._data["label"] == 1]))
		super().CleanData()
	'''
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

