import argparse
from ProcessData import DataCleaner 
import pandas as pd
from DeepNN import DeepNeuralNetwork
import numpy as np
import os
import re
import subprocess
def get_unique_samples(xrootd_path, redirector="root://cmseos.fnal.gov"):
	"""
	xrootd_path: e.g. /store/user/you/parquet/
	"""
	cmd = ["xrdfs", redirector, "ls", xrootd_path]
	result = subprocess.run(cmd, capture_output=True, text=True, check=True)

	pattern = re.compile(
		r"chunk_\d+_sample_(.+?)_type_CMS_Photons\.parquet$"
	)

	samples = set()

	for line in result.stdout.splitlines():
		filename = line.split("/")[-1]
		match = pattern.match(filename)
		if match:
		    samples.add(match.group(1))

	return samples

# DNN for bkg classification (iso vs noniso) 
def runDNN(args):
    #import kerebos credentials to conda env if not already there
    kerb = os.getenv("KRB5CCNAME")
    if(kerb is None):
    	print("Setting kerebos credentials")
    	os.environ["KRB5CCNAME"] = "API:"
    printstats = False
    cleaner = DataCleaner(args.parquetpath, "Photon", "CMS", printstats)
    subdirs = [["JetHT18_RunB","*"],["EGamma18_RunC","*"]] #13281 total EGamma18_RunC chunks - saving 10% of both JetHT18_RunB and EGamma18_RunC for test
    #subdirs = [["JetHT18_RunB",[0,6007]],["EGamma18_RunC",[0,11953]]] #13281 total EGamma18_RunC chunks - saving 10% of both JetHT18_RunB and EGamma18_RunC for test
    if(args.testNetwork is not None):
    	if args.testNetwork == "SMS_GlGl":
    		subdirs = [["JetHT17_RunC","*"],["SMS_GlGl","*"]]
    	elif args.testNetwork == "2017":
    		subdirs = [["JetHT17_RunC","*"],["DoubleEG17_RunC","*"]]
    	else:
    		print("test conditions",args.testNetwork,"does not have an associated dataset with it")
    		exit()
    #subdirs = [["JetHT18_RunB","*"],["JetHT18_RunC","*"],["EGamma18_RunC","*"],["SMS_GlGl","*"]
    blocksize = "100 MB"
    dask_df = cleaner.GetDaskData(subdirs, blocksize, debug=args.debug)
    dask_df_cleaned = cleaner.CleanDaskData(dask_df, dropna = True, do_iso_presel=False)
    cleaner.ConvertToPandas(dask_df_cleaned)
    cleaner.SetPrintStats(True)
    iso_sample = ["EGamma","DoubleEG","SMS_GlGl"]
    cleaner.SelectClass(4,iso_sample)
    cleaner.SelectClass(6,"JetHT")
    if args.endcap:
    	cleaner.EndcapOnly("Photon_EtaCenter")
    else:
    	cleaner.BarrelOnly("Photon_EtaCenter")	
    classes_to_balance = [4,6]
    cleaner.BalanceClasses(classes_to_balance)
    
    shape_cols = ["Photon_EtaSig_CMS","Photon_PhiSig_CMS","Photon_EtaPhiCov_CMS","Photon_majorLength_CMS", "Photon_minorLength_CMS"]
    iso_cols = ["Photon_hcalTowerSumEtConeDR04","Photon_trkSumPtSolidConeDR04","Photon_trkSumPtHollowConeDR04","Photon_hadTowOverEM","Photon_ecalRHSumEtConeDR04"]
    #reader.BalanceClasses([4,6])
    #create sigma columns
    cleaner.MakeSigmas(['Photon_EtaVar_CMS','Photon_PhiVar_CMS'])
    #create new columns of relative isolation
    iso_cols_pt = cleaner.DivideCols(iso_cols,'Photon_Pt_CMS')
    iso_cols = iso_cols_pt
    data = cleaner.GetData()
    
    #labels
    #unmatched = -1
    
    #signal = 0
    #iso bkg = 4
    #iso bkg (gen level) = 5
    #!iso bkg = 6
    
    #phys bkg vs det bkg - these are done in runKUCNN_detector.py
    #phys bkg = 1 
    #BH = 2
    #spike = 3
    
    catToName = {4 : "isoBkg", 6 : "nonIsoBkg"}
    catToColor = {4 : "green", 6 : "red"}
    
    network_name = "KU-DNN_photonID"
    if args.extra is not None:
    	network_name += "_"+args.extra
    nepochs = int(args.nEpochs)
    if args.debug:
    	nepochs = 5	
    early = False
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
    
    #TODO: update args.exclude to take multiple inputs
    if args.exclude is not None:
    	if args.exclude not in cols:
    		print("Feature to exclude",args.exclude,"not currently in selection")
    	else:
    		cols.remove(args.exclude)
    		if "+" in args.exclude:
    			args.exclude = args.exclude.replace("+","p")
    		network_name += "_excludingFeature_"+args.exclude
    
    print("features used",cols)	
    network_name += "_"+str(nepochs)+"epochs"
    if(early):
    	network_name += "_earlyStop"
    
    
    #len(nodes) = # layers
    #nodes[i] = # nodes at ith layer
    network_name += "_MINI_"+args.arch
    if(args.endcap):
    	network_name += "_endcapOnly";

    if(args.arch == "default"):
    	nodes = [64, 64, 64]
    elif(args.arch == "med32"):
    	nodes = [32, 32, 32] 
    elif(args.arch == "med16"):
    	nodes = [16, 16, 16] 
    elif(args.arch == "med8"):
    	nodes = [8, 8, 8] 
    elif(args.arch == "large8"):
    	nodes = [8, 8, 8, 8, 8] 
    elif(args.arch == "small8"):
    	nodes = [8, 8] 
    elif(args.arch == "med4"):
    	nodes = [4, 4, 4] 
    elif(args.arch == "large4"):
    	nodes = [4, 4, 4, 4, 4] 
    elif(args.arch == "small4"):
    	nodes = [4, 4] 
    elif(args.arch == "16-8-4"):
    	nodes = [16, 8, 4] 
    else:
    	print("architecture",args.arch,"not specified")
    	exit()
    
    extra_name = ""
    if(args.testNetwork is not None):
    	extra_name = args.testNetwork
    	if "SMS" in extra_name:
    		extra_name = "SMSasIsoBkg"+extra_name
    	if args.testNetwork == "training_sample": 
    		extra_name = ""
    
    
    model = DeepNeuralNetwork(data,nodes,cols,catToName,catToColor,network_name,extra_name)
    model.BuildModel()
    if(args.testNetwork is not None):
        print("Evaluating network",network_name,"on test sample",subdirs)
        model.VizInputs(extra_name)
        model.TestModel(1,1,ret_fpr_thresh = 0.3)
        print("Evaluating network on external signal sample",network_name)
        return
    if(args.dryRun):
    	exit()
    model.VizInputs()
    model.CompileModel()
    model.summary()
    #input is TrainModel(epochs=1,oname="",int:verb=1)
    model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
    #needs test data + to make ROC plots
    print("Evaluating network on test sample",network_name)
    model.TestModel(1,1,validate_model=True, ret_fpr_thresh = 0.3)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquetpath",help="path to parquet files",required=True)
	parser.add_argument('--network','-n',help="which set of inputs to run",choices=["iso","shape","isoShape"],required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",choices=["default","med16","med8","small8","med32","large8","small4","med4","large4","16-8-4"],default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--exclude",help='exclude feature from training',default=None)
	#parser.add_argument("--reweightClasses",help="reweight classes",default=False,action='store_true')
	parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=None,choices=['SMS_GlGl','2017','training_sample'])
	parser.add_argument('--debug',help='debug mode',default=False,action='store_true')
	parser.add_argument('--endcap',help='train for endcap only',default=False,action='store_true')
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
