import argparse
import os
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
	if redirector in xrootd_path:
		xrootd_path = xrootd_path[xrootd_path.find(redirector)+len(redirector):]
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
	kerb = os.getenv("KRB5CCNAME")
	if(kerb is None):
		print("Setting kerebos credentials")
		os.environ["KRB5CCNAME"] = "API:"
	printstats = False
	cleaner = DataCleaner(args.parquetpath, "SC", args.SCtype, printstats)
	blocksize = args.blocksize
	if(args.testNetwork is None or args.testNetwork == "training_sample"):
		subdirs = []
	elif("SMS" in args.testNetwork):
		subdirs = [["METPD18_RunC","*"],[args.testNetwork,"*"]]
	else:
		print("test scenario",args.testNetwork,"does not have associated test data")
		exit()
	dask_df = cleaner.GetDaskData(subdirs, blocksize=blocksize, debug=args.debug)
	#if("SMS" in args.testNetwork):
	#	dask_df_cleaned = cleaner.CleanDaskData(dask_df)
	#	all_sms_mass_points = set() 
	#	for subdir in subdirs:
	#		if "SMS" not in subdir:
	#			continue
	#		sms_samples = get_unique_samples(args.parquetpath+"/"+subdir)
	#		all_sms_mass_points.update(sms_samples)
	#	sms_samples = get_unique_samples(args.parquetpath+"/"+args.testNetwork)
	#	#cap samples per mass point as to not overwhelm the BH contribution
	#	#can set based on how many mass points there are - ie nsample = 55442 / len(sms_samples)
	#	if "SqSq" in args.testNetwork:
	#		nsample = 100
	#	elif "GlGl" in args.testNetwork:
	#		nsample = 500
	#	else:
	#		nsample = 500
	#	if len(sms_samples) < 1:
	#		dask_df_downsampled = dask_df_cleaned
	#	else:
	#		for sample in sms_samples:
	#			if args.debug and "mGl_1500_mN2_500_mN1_100" not in sample and "GlGl" in args.testNetwork:
	#				continue
	#			if args.debug and "mGl_1700_mN2_1500_mN1_100_ct0p1" not in sample and "SqSq" in args.testNetwork:
	#				continue
	#			dask_df_cleaned = cleaner.CapSampleDask(dask_df_cleaned,sample,500)
	#	cleaner.ConvertToPandas(dask_df_cleaned)
	#else:
	#	cleaner.CleanAndConvert(dask_df)
	cleaner.CleanAndConvert(dask_df)
	cleaner.SelectClass(1,["EGamma","SMS"]) #choose for a certain class (first arg) to only come from sample (second arg)
	cleaner.SelectClass(2,"METPD")
	if not args.addSpikes:
		cleaner.DropClass(3)

	#do preprocessing
	cleaner.BarrelOnly("SC_EtaCenter")

	#set max number of samples with label to be nsamp
	#cleaner.CapClass(3,3000)
	
	#balance classes via random undersampling - default
	catToName = {1 : "physicsBkg", 2 : "beamHalo"}
	catToColor = {1 : "green", 2 : "red", 0 : "pink"}
	classes_to_balance = [1,2]
	if args.addSpikes:
		catToName[3] = "spike"
		catToColor[3] = "orange"
		classes_to_balance.append(3)
	cleaner.BalanceClasses(classes_to_balance)
	cleaner.MakeEnergySum("event_idx","sc_idx","SC_rh_Energy_CMS")
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
	if args.addSpikes:
		network_name += "_withSpikeClass"  
 
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

	extra_name = ""
	if(args.testNetwork is not None):
		extra_name = args.testNetwork
		if "SMS" in extra_name:
			extra_name = "SMSasIsoBkg"+extra_name
		if args.testNetwork == "training_sample": 
			extra_name = ""

	model = ConvNeuralNetwork(data,filters,network_name,args.SCtype,extra_name)
	model.BuildModel()
	model.SetCategoryNames(catToName,catToColor)
	if(args.testNetwork):
		print("Evaluating network",network_name)
		model.VizInputs(extra_name)
		model.TestModel(fpr_threshs = [0.001, 0.002, 0.005, 0.01])
		print("Best model used for testing is",model.GetBestModel())	
		return
	
	#visualize inputs
	model.VizInputs()
	model.CompileModel()
	model.summary()
	if(args.dryRun):
		exit()
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	model.TestModel()
	model.VizModelWeights()
	model.VizFeatureMaps()
	print("Best model used for testing is",model.GetBestModel())	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquetpath",help="path to parquet files",required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument('--addSpikes',help='include spikes in training',action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--SCtype",help='type of SCs to run over',choices=["CMS","BHC","BHCPUCleaned"],default="CMS")
	parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=None,choices=['SMS_GlGl','2017','training_sample'])
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--debug",help="run over only a few parquet files per sample to debug faster",action='store_true',default=False)
	parser.add_argument("--recreatefiles",help='recreate parquet files for training',action='store_true',default=False)
	parser.add_argument("--blocksize",help='chunk size to read parquet files in for dask',default="100 MB")
	args = parser.parse_args()

	runCNN(args)

if __name__ == "__main__":
	main()
