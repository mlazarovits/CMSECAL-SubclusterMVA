import numpy as np
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
		self._tag = ""
		self._output_parquet_data = "parquet_output"
		self._infiles = None
		os.makedirs(self._output_parquet_data,exist_ok=True)
		self._obj = obj

	def SetParquetOutputDir(self, pdir):
		if not os.path.exists(pdir):
			os.mkdir(pdir)
		self._output_parquet_data = pdir
	
class TTreeReader(FileReader):
	def __init__(self, obj, tag, extra="", printStats = False):
		super().__init__(obj, printStats)
		self._tag = tag
		self._output_parquet_data = self._output_parquet_data+f"/{self._tag}_{self._obj}s"
		if extra != "":
			self._output_parquet_data += "_"+extra
		os.makedirs(self._output_parquet_data,exist_ok=True)

	def ProcessCNNBranches(self, file, sample, step_size=10000, debug = False, labelas = -999, recreate_files = False, chunkrange = [-1,-1], dryrun = False):
		branches = [
			f"SC_rh_iEta_{self._tag}",
			f"SC_rh_iPhi_{self._tag}",
			f"SC_rh_Energy_{self._tag}",
			f"SC_trueLabel_{self._tag}",
			f"SC_EtaCenter_{self._tag}",
			f"SC_seedTime_CMS"
		]
		data_accum = []
		nchunk = 0
		total_time = 0

		if(isinstance(step_size,int) and maxnchunk == -1):
			nentries = uproot.open(file)["tree"].num_entries
			nchunks = (nentries + step_size - 1) // step_size
		else:
			nchunks = maxnchunk+1
		ncurrent_files = len([name for name in os.listdir(self._output_parquet_data) if f"sample_{sample}_type_{self._tag}.parquet" in name])
		if os.path.exists(self._output_parquet_data) and not recreate_files and ncurrent_files == nchunks:
			print(nchunks,"parquet files already processed for file",file,"in",self._output_parquet_data,"returning\n")
			return
		print("Processing sample",sample,"from file",file,"with step size",step_size,"and",nchunks,"chunks")
		print("Branches",branches)
		if dryrun:
			print("Dry run only. Returning")
			return
		for chunk in uproot.iterate(file + ":tree", branches, step_size=step_size, library="ak",
				num_workers = 8, #multithreading options
				decompression_executor=ThreadPoolExecutor(max_workers=8),
				interpretation_executor=ThreadPoolExecutor(max_workers=8)
			):
			parquet_fname = f"chunk_{nchunk:05d}_sample_{sample}_type_{self._tag}.parquet"
			if(os.path.exists( os.path.join(self._output_parquet_data, parquet_fname) )) and not recreate_files:
				print(os.path.join(self._output_parquet_data, parquet_fname),"exists - skipping",end="\r",flush=True)
				nchunk += 1
				continue
			t1 = time.perf_counter()
			if nchunk > 2 and debug:
				print("Break from debugging")
				return
			if chunkrange[0] != -1 and nchunk < chunkrange[0]:
				continue
			if chunkrange[1] != -1 and nchunk > chunkrange[1]:
				break
			print(f"Processing chunk #{nchunk}",end="\r",flush=True)
			
			sc_counts = ak.num(chunk[f"SC_trueLabel_{self._tag}"])
			truelabel = ak.to_numpy(ak.flatten(chunk[f"SC_trueLabel_{self._tag}"],axis=1))
			if labelas != -999:
				truelabel = pa.array([labelas] * sum(sc_counts))
			#write to parquet table directly
			table = pa.table({
				"event_idx": np.repeat(np.arange(len(chunk)), sc_counts),
				"sc_idx": ak.to_numpy(ak.flatten(ak.local_index(chunk[f"SC_trueLabel_{self._tag}"]))),
				f"SC_EtaCenter_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"SC_EtaCenter_{self._tag}"],axis=1)),
				f"SC_seedTime_CMS" : ak.to_numpy(ak.flatten(chunk[f"SC_seedTime_CMS"],axis=1)), #ak.to_numpy must be flat arrays
				f"SC_trueLabel_{self._tag}" : truelabel,
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
	
	def ProcessFileCNN(self, file, sample, step_size=10000, chunkrange=[-1,-1], debug=False, labelas = -999, dryrun = False):
		if sample == "" and "SMS" in file:
			match = "_AODSIM_"
			sample = file[file.find("SMS-"):]
			sample = sample.replace("_AODSIM","")
			sample = sample[:sample.find("_superclusters")]
		if sample.find("-") != -1:
			sample = sample.replace("-","_")
		self.ProcessCNNBranches(file,sample,step_size,debug,labelas,chunkrange=chunkrange,dryrun=dryrun)
		print("Wrote parquet chunks to",self._output_parquet_data)

	def ProcessFileDNN(self, file, sample, step_size=10000, chunkrange = [-1,-1], debug=False, labelas = -999, dryrun = False):
		if sample == "" and "SMS" in file:
			match = "_AODSIM_"
			sample = file[file.find("SMS-"):]
			sample = sample.replace("_AODSIM","")
			sample = sample[:sample.find("_photons")]
		if sample.find("-") != -1:
			sample = sample.replace("-","_")
		self.ProcessDNNBranches(file,sample,step_size,debug=debug,labelas=labelas,chunkrange=chunkrange,dryrun=dryrun)
		print("Wrote parquet chunks to",self._output_parquet_data)
	
	def ProcessDNNBranches(self, file, sample, step_size=10000, debug=False, labelas=-999, recreate_files = False, chunkrange=[-1,-1], dryrun = False):
		branches = [
			f"Photon_EtaVar_{self._tag}",
			f"Photon_PhiVar_{self._tag}",
			f"Photon_EtaPhiCov_{self._tag}",
			f"Photon_majorLength_{self._tag}",
			f"Photon_minorLength_{self._tag}",
			f"Photon_hcalTowerSumEtConeDR04",
			f"Photon_trkSumPtSolidConeDR04",
			f"Photon_trkSumPtHollowConeDR04",
			f"Photon_hadTowOverEM",
			f"Photon_ecalRHSumEtConeDR04",
			f"Photon_Pt_{self._tag}",
			f"Photon_EtaCenter_{self._tag}",
			f"Photon_trueLabel_{self._tag}",
			f"PassGJetsCR",
			f"Photon_PassGJetsCR_Obj",
			f"Photon_Energy_{self._tag}"
		]
		if "photons_defaultv4p4" in file:
			branches.append(f"PassDijetsCR")
			branches.append(f"Photon_PassDijetsCR_Obj")
		print("Branches",branches)
		data_accum = []
		nchunk = 0
		total_nchunk = 0	
		total_time = 0

		print("Processing sample",sample,"from file",file,"with step size",step_size)
		tree = uproot.open(file)["tree"]
		nentries = tree.num_entries
		if(isinstance(step_size,int)):
			print("File",file,"has",nentries,"entries and therefore",(nentries + step_size - 1) // step_size,"chunks with step size",step_size)
		else:
			print("Chunking file",file,"in",step_size,"chunks")

		print("chunkrange",chunkrange)
		if dryrun:
			print("Dry run only. Returning")
			return
		for chunk in uproot.iterate(file + ":tree", branches, step_size=step_size, library="ak",
				num_workers = 8, #multithreading options
				decompression_executor=ThreadPoolExecutor(max_workers=8),
				interpretation_executor=ThreadPoolExecutor(max_workers=8)
			):
			print(f"Processing chunk #{nchunk}",end="\r",flush=True)
			parquet_fname = f"chunk_{nchunk:05d}_sample_{sample}_type_{self._tag}_Photons.parquet"
			if(os.path.exists( os.path.join(self._output_parquet_data, parquet_fname) )):
				#print(os.path.join(self._output_parquet_data, parquet_fname),"exists. Please provide a unique sample name for",file)
				#print(os.path.join(self._output_parquet_data, parquet_fname),"exists. Skipping.",end="\r",flush=True)
				print(f"Skipping chunk #{nchunk}",end="\r",flush=True)
				nchunk += 1
				continue
			if nchunk > 2 and debug:
				print("Break from debugging")
				return
			if chunkrange[0] != -1 and nchunk < chunkrange[0]:
				nchunk += 1
				continue
			if chunkrange[1] != -1 and nchunk >= chunkrange[1]:
				break
			t1 = time.perf_counter()
			print(f"Processing chunk #{nchunk}",end="\r",flush=True)

				
			pho_counts = ak.num(chunk[f"Photon_trueLabel_{self._tag}"])
			truelabel = ak.to_numpy(ak.flatten(chunk[f"Photon_trueLabel_{self._tag}"],axis=1))
			if labelas != -999:
				truelabel = pa.array([labelas] * sum(pho_counts))
			pho_counts = ak.num(chunk[f"Photon_trueLabel_{self._tag}"])


			#write to parquet table directly
			table = pa.table({
				"event_idx": np.repeat(np.arange(len(chunk)), pho_counts),
				"pho_idx": ak.to_numpy(ak.flatten(ak.local_index(chunk[f"Photon_trueLabel_{self._tag}"]))),
				f"Photon_trueLabel_{self._tag}" : truelabel,
				f"Photon_EtaVar_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"Photon_EtaVar_{self._tag}"],axis=1)),
				f"Photon_PhiVar_{self._tag}" : ak.to_numpy(ak.flatten(chunk[f"Photon_PhiVar_{self._tag}"],axis=1)),
				f"Photon_EtaPhiCov_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_EtaPhiCov_{self._tag}"],axis=1)),
				f"Photon_majorLength_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_majorLength_{self._tag}"],axis=1)),
				f"Photon_minorLength_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_minorLength_{self._tag}"],axis=1)),
				f"Photon_hcalTowerSumEtConeDR04": ak.to_numpy(ak.flatten(chunk[f"Photon_hcalTowerSumEtConeDR04"],axis=1)),
				f"Photon_trkSumPtSolidConeDR04": ak.to_numpy(ak.flatten(chunk[f"Photon_trkSumPtSolidConeDR04"],axis=1)),
				f"Photon_trkSumPtHollowConeDR04": ak.to_numpy(ak.flatten(chunk[f"Photon_trkSumPtHollowConeDR04"],axis=1)),
				f"Photon_hadTowOverEM": ak.to_numpy(ak.flatten(chunk[f"Photon_hadTowOverEM"],axis=1)),
				f"Photon_ecalRHSumEtConeDR04": ak.to_numpy(ak.flatten(chunk[f"Photon_ecalRHSumEtConeDR04"],axis=1)),
				f"Photon_Pt_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_Pt_{self._tag}"],axis=1)),
				f"Photon_Energy_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_Energy_{self._tag}"],axis=1)),
				f"Photon_EtaCenter_{self._tag}": ak.to_numpy(ak.flatten(chunk[f"Photon_EtaCenter_{self._tag}"],axis=1)),
				"sample": pa.array([sample] * sum(pho_counts))
			})	
			if "PassDijetsCR" in branches:
				passdijetscr, _ = ak.broadcast_arrays(chunk["PassDijetsCR"], chunk[f"Photon_trueLabel_{self._tag}"])
				table["PassDijetsCR"] = passdijetscr
				passdijetscr_obj = ak.to_numpy(ak.flatten(chunk[f"Photon_PassDijetsCR_Obj"],axis=1))
				table["Photon_PassDijetsCR_Obj"] = passdijetscr_obj
				table["Photon_PassGJetsCR_Obj"] = ak.to_numpy(ak.flatten(chunk[f"Photon_PassGJetsCR_Obj"],axis=1))
			pq.write_table(table, os.path.join(self._output_parquet_data, parquet_fname))
			#data_accum.append(df)
			nchunk += 1
			total_nchunk += 1
			t2 = time.perf_counter()
			total_time += (t2 - t1)
			print(f"Chunk #{nchunk} processed took",(t2-t1),"seconds",end="\r",flush = True)
		if nchunk == 0:
			print("Ran over 0 chunks")
		else:
			print("Done processing file",file,"took",total_time,"seconds total with",total_time / nchunk,"seconds on average per chunk with",total_nchunk,"chunks\n\n")
	

	def ReadDataFromParquetTable(self):
		dataset = pq.ParquetDataset(self._output_parquet_data)
		table = dataset.read()
		return table.to_pandas()	


