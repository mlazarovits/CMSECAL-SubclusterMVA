import argparse
from ProcessData import DataCleaner
import pandas as pd
from DeepNN import DeepNeuralNetwork
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
def runDNN(args):
	#get names of mass points for sample balancing
	remote_path_sms = args.parquetpath
	remote_path_sms = remote_path_sms[remote_path_sms.rfind("//")+1:]
	subdirs = ["JetHT18_RunC","SMS_"+args.subproc]
	all_sms_mass_points = set() 
	for subdir in subdirs:
		if "SMS" not in subdir:
			continue
		sms_samples = get_unique_samples(remote_path_sms+"/"+subdir)
		all_sms_mass_points.update(sms_samples)
	sms_samples = get_unique_samples(remote_path_sms+"/SMS_"+args.subproc)
	
	printstats = False
	cleaner = DataCleaner(args.parquetpath, "Photon", args.SCtype, printstats)
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
			dask_df_downsampled = cleaner.CapSampleDask(dask_df_cleaned,sample,500)
	cleaner.ConvertToPandas(dask_df_downsampled)
	cleaner.SetPrintStats(True)
	cleaner.DropClass(3)
	#test with SMS (not seen) and unseen MET PD for BH
	cleaner.SelectClass(1,"SMS"); #choose for a certain class (first arg) to only come from sample (second arg)
	cleaner.SelectClass(2,"METPD18_RunC")

	#do preprocessing
	cleaner.BarrelOnly("Photon_EtaCenter")

	#balance classes via random undersampling - default
	catToName = {4 : "isoBkg", 6 : "nonIsoBkg"}
	catToColor = {4 : "green", 6 : "red"}
	classes_to_balance = [4,6]
	cleaner.BalanceClasses(classes_to_balance)
	cleaner.MakeSigmas(['Photon_EtaVar_CMS','Photon_PhiVar_CMS'])
	data = cleaner.GetData()

	
	shape_cols = ["Photon_EtaSig_CMS","Photon_PhiSig_CMS","Photon_EtaPhiCov_CMS","Photon_majorLength_CMS", "Photon_minorLength_CMS"]
	iso_cols = ["Photon_hcalTowerSumEtConeDR04","Photon_trkSumPtSolidConeDR04","Photon_trkSumPtHollowConeDR04","Photon_hadTowOverEM","Photon_ecalRHSumEtConeDR04"]
	network_name = "KU-DNN_photonID"
	if args.extra is not None:
		network_name += "_"+args.extra
	nepochs = int(args.nEpochs)
	if(args.network == "shape"):
		#default input set
		cols = shape_cols 
	elif(args.network == "iso"):
		#default input set
		cols = iso_cols 
	elif(args.network == "isoShape"):
		cols = shape_cols + iso_cols
	else:
		print("Invalid network selected",args.network)
	network_name += "_"+args.network


	print("features used",cols)	
	network_name += "_"+str(nepochs)+"epochs"
	
	
	#len(nodes) = # layers
	#nodes[i] = # nodes at ith layer
	network_name += "_"+args.arch
	arch_map = {}
	arch_map["default"] = [64, 64, 64]
	arch_map["med16"] = [16, 16, 16]
	arch_map["med8"] = [8, 8, 8]
	arch_map["small8"] = [8, 8]
	arch_map["large8"] = [8, 8, 8, 8, 8]

	nodes = arch_map[args.arch]	
	
	model = DeepNeuralNetwork(None,nodes,cols,catToName,catToColor,network_name,"SMSasIsoBkg_"+args.subproc)
	model.SetTestData(data)
	model.BuildModel()
	model.CompileModel()
	model.summary()
	print("Evaluating network on test sample",network_name)
	model.TestModel(1,1,ret_fpr_thresh = 0.3)
	print("Evaluating network on external signal sample",network_name)
	print("Best model used for testing is",model.GetBestModel())	
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquetpath",help="path to parquet files",required=True)
	parser.add_argument('--network','-n',help="which set of inputs to run",choices=["iso","shape","isoShape"],required=True)
	parser.add_argument("--subproc",help="subprocess",choices=["GlGl","SqSq","GlGlZ"],required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",required=True)
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--SCtype",help='type of SCs to run over',choices=["CMS","BHC","BHCPUCleaned"],default="CMS")
	#ONLY DOES TEST
	#parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=False,action='store_true')
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--debug",help="run over only a few parquet files per sample to debug faster",action='store_true',default=False)
	parser.add_argument("--blocksize",help='chunk size to read parquet files in for dask',default="100 MB")
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
