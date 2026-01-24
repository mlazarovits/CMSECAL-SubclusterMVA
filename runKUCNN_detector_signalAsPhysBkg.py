import argparse
from ProcessData import DataCleaner
import pandas as pd
from ConvNN import ConvNeuralNetwork
import numpy as np
import subprocess
import re

def get_unique_samples(xrootd_path, redirector="root://cmseos.fnal.gov"):
	"""
	xrootd_path: e.g. /store/user/you/parquet/
	"""
	cmd = ["xrdfs", redirector, "ls", xrootd_path]
	result = subprocess.run(cmd, capture_output=True, text=True, check=True)

	pattern = re.compile(
	    r"chunk_\d+_sample_(.+?)_type_CMS\.parquet$"
	)

	samples = set()

	for line in result.stdout.splitlines():
	    filename = line.split("/")[-1]
	    match = pattern.match(filename)
	    if match:
	        samples.add(match.group(1))

	return samples



# CNN for identifying detector background (spikes + beam halo) from physics bkg
def runCNN(args):
	#get names of mass points for sample balancing
	remote_path_sms = args.parquetpath
	remote_path_sms = remote_path_sms[remote_path_sms.rfind("//")+1:]	
	subdirs = ["METPD18_RunC","SMS_"+args.subproc]
	all_sms_mass_points = set() 
	for subdir in subdirs:
		if "SMS" not in subdir:
			continue
		sms_samples = get_unique_samples(remote_path_sms+"/"+subdir)
		all_sms_mass_points.update(sms_samples)
	sms_samples = get_unique_samples(remote_path_sms+"/SMS_"+args.subproc)
	
	printstats = False
	cleaner = DataCleaner(args.parquetpath, "SC", args.SCtype, printstats)
	blocksize = args.blocksize
	dask_df = cleaner.GetDaskData(subdirs, blocksize, args.debug)
	dask_df_cleaned = cleaner.CleanDaskData(dask_df)
	
	#cap samples per mass point as to not overwhelm the BH contribution
	#can set based on how many mass points there are - ie nsample = 55442 / len(sms_samples)
	if args.subproc == "SqSq":
		nsample = 100
	elif args.subproc == "GlGl":
		nsample = 500
	else:
		nsample = 500
	if len(sms_samples) < 1:
		dask_df_downsampled = dask_df_cleaned
	else:
		for sample in sms_samples:
			if args.debug and "mGl_1500_mN2_500_mN1_100" not in sample and "GlGl" in args.subproc:
				continue
			if args.debug and "mGl_1700_mN2_1500_mN1_100_ct0p1" not in sample and args.subproc == "SqSq":
				continue
			dask_df_cleaned = cleaner.CapSampleDask(dask_df_cleaned,sample,500)
	cleaner.ConvertToPandas(dask_df_cleaned)
	cleaner.SetPrintStats(True)
	cleaner.DropClass(3)
	#test with SMS (not seen) and unseen MET PD for BH
	cleaner.SelectClass(1,"SMS"); #choose for a certain class (first arg) to only come from sample (second arg)
	cleaner.SelectClass(2,"METPD18_RunC")

	#do preprocessing
	cleaner.BarrelOnly("SC_EtaCenter")

	#balance classes via random undersampling - default
	catToName = {1 : "physicsBkg", 2 : "beamHalo"}
	catToColor = {1 : "green", 2 : "red", 0 : "pink"}
	classes_to_balance = [1,2]
	cleaner.BalanceClasses(classes_to_balance)
	data = cleaner.GetData()

	
	#"features" to use for training
	#for a CNN this is just a weighted map of the subclusters in eta-phi 2D space	
	network_name = "KU-CNN_detector"
	if args.extra is not None:
		network_name += "_"+args.extra
	nepochs = int(args.nEpochs)
	early = False
	network_name += "_"+str(nepochs)+"epochs"
	if(early):
		network_name += "_earlyStop"
 
	network_name += "_"+args.arch
	arch_map = {}
	arch_map["default"] = [64, 64, 64] 
	arch_map["xsmall3"] = [3, 3] 
	arch_map["small2"] = [2, 2, 2] 
	arch_map["small3"] = [3, 3, 3] 
	arch_map["small4"] = [4, 4, 4] 
	arch_map["small8"] = [8, 8, 8] 
	arch_map["8_4_2"] = [8, 4, 2]
	arch_map["16_8_2"]  = [16, 8, 2]
	arch_map["16_8_4_2"]  = [16, 8, 4, 2]
	arch_map["3HalfTallHalfLong"] = ["3HalfTallHalfLong"]
	#tall = (3,2) #[5,2]
	#tall_filters = [tall] * nfilters
	#long = (2,3) #[2,5]
	#arch_map["tallLong_8_4_2"]
	 
	if args.arch not in arch_map.keys():
		print("Invalid architecture selected",args.network)
		print("Available architectures are",arch_map.keys())
		exit()
	
	filters = arch_map[args.arch] 
	
	model = ConvNeuralNetwork(None,filters,network_name,args.SCtype,"SMSasPhysBkg_"+args.subproc)
	#TODO - add string to args of SetTestData to append to figures, lines in txt file, etc
	model.SetTestData(data)
	model.BuildModel()
	model.SetCategoryNames(catToName,catToColor)
	print("Evaluating network",network_name)
	model.summary()
	model.TestModel()
	print("Best model used for testing is",model.GetBestModel())	
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquetpath",help="path to parquet files",required=True)
	parser.add_argument("--subproc",help="subprocess",choices=["GlGl","SqSq","GlGlZ"],required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--SCtype",help='type of SCs to run over',choices=["CMS","BHC","BHCPUCleaned"],default="CMS")
	#ONLY DOES TEST
	#parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=False,action='store_true')
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--debug",help="run over only a few parquet files per sample to debug faster",action='store_true',default=False)
	parser.add_argument("--blocksize",help='chunk size to read parquet files in for dask',default="100 MB")
	args = parser.parse_args()

	runCNN(args)

if __name__ == "__main__":
	main()
