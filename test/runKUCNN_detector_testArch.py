import argparse
from ProcessData import DataCleaner
import pandas as pd
from ConvNN import ConvNeuralNetwork
from dualConvNN import dualConvNeuralNetwork
import numpy as np
import os

def make_arch_map():
	arch_map = {}
	arch_map["default"] = [64, 64, 64] 
	arch_map["xsmall3"] = [3, 3] 
	arch_map["small2"] = [2, 2, 2] 
	arch_map["small3"] = [3, 3, 3] 
	arch_map["small3-2"] = [3, 3, 2] 
	arch_map["small4"] = [4, 4, 4] 
	arch_map["small8"] = [8, 8, 8] 
	arch_map["8_4_2"] = [8, 4, 2]
	arch_map["16_8_4"]  = [16, 8, 4]
	arch_map["3HalfTallHalfLong"] = ["3HalfTallHalfLong"]
	arch_map["3HalfTallHalfLongMultiply"] = ["3HalfTallHalfLongMultiply"]
	#tall = (3,2) #[5,2]
	#tall_filters = [tall] * nfilters
	#long = (2,3) #[2,5]
	#arch_map["tallLong_8_4_2"]
	return arch_map

def make_model(args, data, arch, extra): 
	#"features" to use for training
	#for a CNN this is just a weighted map of the subclusters in eta-phi 2D space	
	network_name = "KU-CNN_detector"
	if extra is not None:
		network_name += "_"+extra
	nepochs = int(args.nEpochs)
	if args.debug:
		nepochs = 5
	early = False
	network_name += "_"+str(nepochs)+"epochs"
	if(early):
		network_name += "_earlyStop"
	if args.addSpikes:
		network_name += "_withSpikeClass"  

	arch_map = make_arch_map()
	filters = arch_map[arch] 
	network_name += "_arch"+arch
	#don't redo training
	if os.path.exists("results/"+network_name+"_"+args.SCtype) and not args.testNetwork:
		print("results/"+network_name,"exists - skipping this training")
		return
	

	
	catToName = {1 : "physicsBkg", 2 : "beamHalo"}
	catToColor = {1 : "green", 2 : "red", 0 : "pink"}
	if args.addSpikes:
		catToName[3] = "spike"
		catToColor[3] = "orange"


	fprs_to_test = [0.05, 0.01, 0.005, 0.001]
	print("fprs_to_test",fprs_to_test)	
	model = ConvNeuralNetwork(data,filters,network_name,args.SCtype)
	model.BuildModel()
	model.SetCategoryNames(catToName,catToColor)
	viz = True
	if args.debug:
		viz = False
	if(args.testNetwork):
		print("Evaluating network",network_name)
		model.summary()
		model.TestModel(viz=viz, batch_size=1, verb=1, validate_model=False, fpr_threshs=fprs_to_test)
		model.VizModelWeights()
		model.VizFeatureMaps()
		print("Best model used for testing is",model.GetBestModel())	
		return
	
	#visualize inputs
	model.VizInputs()
	model.CompileModel()
	model.summary()
	if(args.dryRun):
		return
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	model.TestModel(viz=viz, batch_size=1, verb=1, validate_model=False, fpr_threshs=fprs_to_test)
	model.VizModelWeights()
	model.VizFeatureMaps()
	print("Best model used for testing is",model.GetBestModel(),"for architecture",network_name)

# CNN for identifying detector background (spikes + beam halo) from physics bkg
def runCNNs(args):
	printstats = True
	cleaner = DataCleaner(args.parquetpath, "SC", args.SCtype, printstats)
	dask_df = cleaner.GetDaskData(debug=args.debug)
	cleaner.CleanAndConvert(dask_df)
	cleaner.SelectClass(1,"EGamma"); #choose for a certain class (first arg) to only come from sample (second arg)
	if not args.addSpikes:
		cleaner.DropClass(3)

	#do preprocessing
	cleaner.BarrelOnly("SC_EtaCenter")

	#set max number of samples with label to be nsamp
	#cleaner.CapClass(3,3000)
	
	#balance classes via random undersampling - default
	classes_to_balance = [1,2]
	if args.addSpikes:
		classes_to_balance.append(3)
	cleaner.BalanceClasses(classes_to_balance)
	data = cleaner.GetData()
	data1 = data
	
	#3, 3, 2
	#same as before just dual class instead of multi
	arch = "small3-2"
	print("Doing model with arch",arch)
	make_model(args, data, arch, args.extra)
	print()
	print()
	
	#3, 3
	#same as before just dual class instead of multi
	arch = "xsmall3"
	print("Doing model with arch",arch)
	make_model(args, data, arch, args.extra)
	print()
	print()
	exit()
	
	#3, 3, 3
	#same as before just dual class instead of multi
	arch = "small3"
	print("Doing model with arch",arch)
	make_model(args, data, arch, args.extra)
	print()
	print()
	
	#8, 4, 2
	arch = "8_4_2"
	print("Doing model with arch",arch)
	make_model(args, data1, arch, args.extra)
	print()
	print()

	#16, 8, 2
	arch = "16_8_4"
	print("Doing model with arch",arch)
	make_model(args, data, arch, args.extra)
	print()
	print()

	#mixed filter sizes
	arch = "3HalfTallHalfLong"
	print("Doing model with arch",arch)
	make_model(args, data, arch, args.extra)
	print()
	print()
	
	#arch = "3HalfTallHalfLongMultiply"
	#print("Doing model with arch",arch)
	#make_model(args, data, arch, args.extra)
	#print()
	#print()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquetpath",help="path to parquet files",required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument('--addSpikes',help='include spikes in training',action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--SCtype",help='type of SCs to run over',choices=["CMS","BHC","BHCPUCleaned"],default="CMS")
	parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=False,action='store_true')
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--debug",help="run over only a few parquet files per sample to debug faster",action='store_true',default=False)
	parser.add_argument("--recreatefiles",help='recreate parquet files for training',action='store_true',default=False)
	args = parser.parse_args()

	runCNNs(args)

if __name__ == "__main__":
	main()
