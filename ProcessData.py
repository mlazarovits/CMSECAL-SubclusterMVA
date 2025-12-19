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

class DataCleaner:
	def __init__(self, parquet_path, obj, objtype, printStats = True):
		self._data = None
		self._printstats = printStats
		self._output_parquet_data = parquet_path
		self._obj = obj
		self._tag = objtype
			
	def GetDaskData(self, subdirs = [], blocksize = "100 MB", debug = False):
		print("Reading from parquet files at",self._output_parquet_data,"into Dask df")
		t1 = time.perf_counter()
		if self._output_parquet_data[-1] != "/":
			self._output_parquet_data += "/"
		if len(subdirs) < 1: 
			print("reading parquet files in",self._output_parquet_data)
			if debug:
				print("Debug mode")
				parquet_files = self._output_parquet_data+"chunk_00000_sample_*.parquet"
			else:
				parquet_files = self._output_parquet_data+"*.parquet"
		else:
			parquet_files = []
			for subdir in subdirs:
				print("reading parquet files in",self._output_parquet_data+"/"+subdir)
				if debug:
					print("Debug mode")
					if "SMS_GlGl" in subdir:
						parquet_files.append(self._output_parquet_data+"/"+subdir+"/chunk_00000_*GlGl_mGl_1500_mN2_500_mN1_100*.parquet")
					else:
						parquet_files.append(self._output_parquet_data+"/"+subdir+"/chunk_00000_sample_*.parquet")	
				else:
					parquet_files.append(self._output_parquet_data+"/"+subdir+"/*.parquet")
		print("parquet_files",parquet_files)	
		ddf = dd.read_parquet(parquet_files,blocksize=blocksize)
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to read data from parquet table, total # rows",ddf.shape[0].compute())
		if ddf.shape[0].compute() == 0:
			print(ddf.shape[0].compute(),"rows in parquet data, exiting")
			exit()
		#rename cols
		new_cols = ["label" if col == f"{self._obj}_trueLabel_{self._tag}" else col for col in ddf.columns]
		new_cols = [f"{self._obj}_EtaCenter" if col == f"{self._obj}_EtaCenter_{self._tag}" else col for col in new_cols]
		new_cols = [f"{self._obj}_seedTime" if col ==  f"{self._obj}_seedTime_CMS" else col for col in new_cols]
		ddf = ddf.rename(columns=dict(zip(ddf.columns, new_cols))) #inplace not supported for dask dfs!! (lazy execution remember??)
		self.PrintStatsDask(ddf)
		return ddf

	def SetPrintStats(self, p):
		self._printstats = p

	#cleans data
	def CleanDaskData(self, ddf, dropna = False):
		print("cleaning dask data")
		t1 = time.perf_counter()
		"""
		Cleans the input DataFrame or Dask DataFrame:
		- removes invalid/unmatched labels
		- applies column cuts (e.g., Energy)
		- drops rows with NaNs
		Returns a cleaned Dask DataFrame (lazy until compute()).
		"""


		#  Print initial stats (compute row count lazily)
		if self._printstats:
			nrows = ddf.shape[0].compute()  # works for Dask
			print("Cleaning data", nrows, self._obj+"s","initially")
			self.PrintStatsDask(ddf)
				
		#  Remove invalid labels (lazy, memory-efficient)
		ddf = ddf.query("label != -1 and label != -999")
		if self._printstats:
			print("Total after unmatched/invalid label removal:", ddf.shape[0].compute())
			self.PrintStatsDask(ddf)

		#  Apply column cuts (e.g., Energy)
		if "Energy" in ddf.columns:
			print("Applying energy cut")
			# Ensure ApplyColCut works with Dask: avoid .values
			ddf = self.ApplyColCut("Energy", 30, ddf)
			if prinstats:
				print("Total after energy cut > 30:",ddf.shape[0].compute() )
				self.PrintStatsDask(ddf)
	
		#  Drop rows with any NaNs
		# Lazy operation; assign back
		if dropna:
			ddf = ddf.dropna(how="any")
		
		# check for NaNs per column (compute only scalars) - takes too long
		#for col in ddf.columns:
		#	has_nan = ddf[col].isna().any().compute()  # lazy scalar
		#	if has_nan:
		#		print("Warning: column", col, "still has NaNs")
	
		if self._printstats:
			print("Total after dropna:",ddf.shape[0].compute())
			self.PrintStatsDask(ddf)
	
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to clean dask data")
		return ddf


	def ConvertToPandas(self, ddf):
		self._data = ddf.compute() 
		print("ConvertToPandas - len self data",len(self._data))

	#cleans data and converts to pandas
	def CleanAndConvert(self, ddf):
		print("cleaning dask data")
		t1 = time.perf_counter()
		"""
		Cleans the input DataFrame or Dask DataFrame:
		- removes invalid/unmatched labels
		- applies column cuts (e.g., Energy)
		- drops rows with NaNs
		Returns a cleaned Dask DataFrame (lazy until compute()).
		"""


		#  Print initial stats (compute row count lazily)
		if self._printstats:
			nrows = ddf.shape[0].compute()  # works for Dask
			print("Cleaning data", nrows, self._obj+"s","initially")
			self.PrintStatsDask(ddf)
				
		#  Remove invalid labels (lazy, memory-efficient)
		ddf = ddf.query("label != -1 and label != -999")
		if self._printstats:
			print("Total after unmatched/invalid label removal:", ddf.shape[0].compute())
			self.PrintStatsDask(ddf)

		#  Apply column cuts (e.g., Energy)
		if "Energy" in ddf.columns:
			print("Applying energy cut")
			# Ensure ApplyColCut works with Dask: avoid .values
			ddf = self.ApplyColCut("Energy", 30, ddf)
			if prinstats:
				print("Total after energy cut > 30:",ddf.shape[0].compute() )
				self.PrintStatsDask(ddf)
	
		#  Drop rows with any NaNs
		# Lazy operation; assign back
		ddf = ddf.dropna(how="any")

		# check for NaNs per column (compute only scalars) - takes too long
		#for col in ddf.columns:
		#	has_nan = ddf[col].isna().any().compute()  # lazy scalar
		#	if has_nan:
		#		print("Warning: column", col, "still has NaNs")
	
		if self._printstats:
			print("Total after dropna:",ddf.shape[0].compute())
			self.PrintStatsDask(ddf)
	
		t2 = time.perf_counter()
		print("took",(t2-t1),"seconds to clean dask data")
		self._data = ddf.compute()

	def PrintStatsDask(self, ddf):
		phys = (ddf["label"] == 1).sum().compute()
		if(phys > 0):
			BH = (ddf["label"] == 2).sum().compute()
			spike = (ddf["label"] == 3).sum().compute()
			tot = phys + spike + BH 
			print(" ",tot, ("phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
			return

		isobkg = (ddf["label"] == 4).sum().compute()
		if(isobkg > 0):
			nonisobkg = (ddf["label"] == 6).sum().compute()
			tot = isobkg + nonisobkg
			print(" ",tot, ("isobkg: "+str(isobkg)+" {:.2f}%, nonisobkg: "+str(nonisobkg)+" {:.2f}%").format(isobkg/tot,nonisobkg/tot))

	def PrintStats(self, data = None):
		if data is None:
			data = self._data
		phys = len(data[data["label"] == 1])
		BH = len(data[data["label"] == 2])
		spike = len(data[data["label"] == 3])
		tot = phys + BH + spike
		if(phys > 0):
			print(" ",tot, ("phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
			return

		isobkg = len(data[data["label"] == 4])
		nonisobkg = len(data[data["label"] == 6])
		tot = len(data)
		if(isobkg > 0):
			print(" ",tot, ("isobkg: "+str(isobkg)+" {:.2f}%, nonisobkg: "+str(nonisobkg)+" {:.2f}%").format(isobkg/tot,nonisobkg/tot))
	
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
		mask = (self._data["label"] == nclass) | (~self._data["sample"].str.contains(samp))
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
	
	def CapSample(self,sample,nsamp):
		sampled_subset = self._data[self._data['sample'] == sample].sample(n=nsamp, random_state=42)	
		#replace all rows with label l by this sampled subset
		self._data = pd.concat([self._data[self._data['sample'] != sample], sampled_subset], ignore_index=True)

	def CapSampleDask(self, ddf, sample, nsamp):
		# rows matching sample
		ddf_sample = ddf[ddf["sample"] == sample]
		ddf_other  = ddf[ddf["sample"] != sample]
	
		# total rows in this sample (lazy)
		total = ddf_sample.shape[0].compute()
	
		if total <= nsamp:
		    return ddf
	
		frac = nsamp / total
	
		sampled_subset = ddf_sample.sample(
		    frac=frac,
		    random_state=42
		)
	
		ret_ddf = dd.concat(
		    [ddf_other, sampled_subset],
		    interleave_partitions=True
		)
		if self._printstats:
			print("Total after balancing sample",sample+": ", ret_ddf.shape[0].compute())
			self.PrintStatsDask(ret_ddf)
		return ret_ddf


	
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
	

	def DropClass(self, label):
		self._data = self._data[self._data["label"] != label]
		if(self._printstats):
			print("After dropping class",label)
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
			newcolname = col[:col.find("Var")]+"Sig_"+self._tag
			self._data[newcolname] = np.sqrt(self._data[col])
		print("make sigmas - all cols",self._data.columns)
	
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
		if etabranch not in self._data.columns:
			print(etabranch,"not in data columns",self._data.columns)
			exit()
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

