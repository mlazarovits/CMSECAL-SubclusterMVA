import argparse
from ProcessData import DataCleaner 
import pandas as pd
from DeepNN import DeepNeuralNetwork
import numpy as np
import os

# DNN for bkg classification (iso vs noniso) 
def runDNN(args):
	#import kerebos credentials to conda env if not already there
	kerb = os.getenv("KRB5CCNAME")
	if(kerb is None):
		print("Setting kerebos credentials")
		os.environ["KRB5CCNAME"] = "API:"
	printstats = False
	cleaner = DataCleaner(args.parquetpath, "Photon", "CMS", printstats)
	subdirs = [["JetHT18_RunB","*"],["JetHT18_RunC","*"],["EGamma18_RunC","*"]]
	if args.endcap: #take only subset for endcap since the stats aren't that bad without iso presel
		subdirs = [["JetHT18_RunB",[0,100]],["JetHT18_RunC",[0,100]],["EGamma18_RunC","*"]]
	#subdirs = [["JetHT18_RunB","*"],["JetHT18_RunC","*"],["EGamma18_RunC","*"],["SMS_GlGl","*"]
	blocksize = "100 MB"
	do_iso_presel = True
	if args.endcap:
		do_iso_presel = False
	dask_df = cleaner.GetDaskData(subdirs, blocksize, debug=args.debug)
	dask_df_cleaned = cleaner.CleanDaskData(dask_df, do_iso_presel=do_iso_presel)
	cleaner.ConvertToPandas(dask_df_cleaned)
	cleaner.SetPrintStats(True)
	cleaner.SelectClass(4,"EGamma")
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
	network_name += "_"+args.arch
	if(args.endcap):
		network_name += "_endcapOnly";
	if(args.arch == "default"):
		nodes = [64, 64, 64]
	if(args.arch == "med32"):
		nodes = [32, 32, 32] 
	if(args.arch == "med16"):
		nodes = [16, 16, 16] 
	if(args.arch == "med8"):
		nodes = [8, 8, 8] 
	if(args.arch == "large8"):
		nodes = [8, 8, 8, 8, 8] 
	if(args.arch == "small8"):
		nodes = [8, 8] 


	
	model = DeepNeuralNetwork(data,nodes,cols,catToName,catToColor,network_name)
	model.VizInputs()
	model.BuildModel()
	model.CompileModel()
	model.summary()
	if(args.testNetwork):
	    print("Evaluating network on test sample",network_name)
	    model.TestModel(1,1,ret_fpr_thresh = 0.3)
	    print("Evaluating network on external signal sample",network_name)
	    return
	if(args.dryRun):
		exit()
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	print("Evaluating network on test sample",network_name)
	model.TestModel(1,1,validate_model=True, ret_fpr_thresh = 0.3)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquetpath",help="path to parquet files",required=True)
	parser.add_argument('--network','-n',help="which set of inputs to run",choices=["iso","shape","isoShape"],required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",choices=["default","med16","med8","small8","med32","large8"],default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--exclude",help='exclude feature from training',default=None)
	#parser.add_argument("--reweightClasses",help="reweight classes",default=False,action='store_true')
	parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=False,action='store_true')
	parser.add_argument('--debug',help='debug mode',default=False,action='store_true')
	parser.add_argument('--endcap',help='train for endcap only',default=False,action='store_true')
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
