import argparse
from ProcessData import CSVReader
import pandas as pd
import numpy as np
from DeepNN import DeepNeuralNetwork

# DNN for bkg classification (iso vs noniso) 
def runDNN(args):
	#data
	printstats = True
	reader = TTreeReader("Photon",printstats)
	reader.AddFileDNN("")
	
	shape_cols = ["EtaSig","PhiSig","EtaPhiCov","majorLength", "minorLength"]
	iso_cols = ["hcalTowerSumEtConeDR04","trkSumPtSolidConeDR04","trkSumPtHollowConeDR04","hadTowOverEM","ecalRHSumEtConeDR04"]
	reader.CleanDataDask()
	reader.BarrelOnly("Photon_EtaCenter")
	reader.BalanceClasses([4,6])
	reader.MakeSigmas(['EtaVar','PhiVar'])
	#create new columns of relative isolation
	iso_cols_pt = reader.DivideCols(iso_cols,'Pt')
	data = reader.GetData()

	'''
	#process SMS data - will add test nonisobkg data after train_test_split in DeepNN ctor
	SMSreader = CSVReader(printstats)
	SMSreader.AddFile("csv/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-500_photons_defaultv3p10_noIso_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")
	SMSreader.MakeSigmas(['EtaVar','PhiVar'])
	SMSreader.DivideCols(iso_cols,'Pt')
	#only want photons from n2 (label 0) and non iso bkg (label 6)
	SMSreader.RemoveEntries('label',1)
	SMSreader.RemoveEntries('label',5)
	#switch label 0 to label 4 to act as 'iso bkg' sig
	SMSreader.SetFeatureFromValToVal('label',0,4)
	SMSreader.BarrelOnly()
	SMSreader.CleanData()
	#SMSreader.CapClass(4,3000)
	#SMSreader.BalanceClasses([4,6])
	sms_data = SMSreader.GetData()
	'''
	iso_cols = iso_cols_pt

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

	default_cols = ["sample","event","object","label"]
	network_name = "KU-DNN_photonID"
	if args.extra is not None:
		network_name += "_"+args.extra
	nepochs = int(args.nEpochs)
	early = False
	if(args.network == "shape"):
		#default input set
		shape_cols += default_cols
		cols = shape_cols 
	elif(args.network == "iso"):
		#default input set
		iso_cols += default_cols
		cols = iso_cols 
	elif(args.network == "isoShape"):
		cols = default_cols + shape_cols + iso_cols
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

	if(args.dryRun):
		cols.append("Energy")
	#if(args.reweightClasses):
	#	cols.append("weight")

	print("features used",cols)	
	network_name += "_"+str(nepochs)+"epochs"
	if(early):
		network_name += "_earlyStop"
	
	
	#len(nodes) = # layers
	#nodes[i] = # nodes at ith layer
	network_name += "_"+args.arch
	if(args.arch == "default"):
		nodes = [64, 64, 64]
	if(args.arch == "med16"):
		nodes = [16, 16, 16] 
	if(args.arch == "med8"):
		nodes = [8, 8, 8] 
	if(args.arch == "small8"):
		nodes = [8, 8] 
	
	
	model = DeepNeuralNetwork(data,nodes,cols,catToName,catToColor,network_name)
	model.BuildModel()
	if(args.testNetwork):
	    print("Evaluating network on test sample",network_name)
	    model.TestModel(1,1)
	    print("Evaluating network on external signal sample",network_name)
	    model.TestModel_ExternalSignal(sms_data,cols)
	    return
	model.CompileModel()
	model.summary()
	if(args.dryRun):
		exit()
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	print("Evaluating network on test sample",network_name)
	model.TestModel(1,1,validate_model=True)
	print("Evaluating network on external signal sample",network_name)
	model.TestModel_ExternalSignal(sms_data,cols)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--network','-n',help="which set of inputs to run",choices=["iso","shape","isoShape"],required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",choices=["default","med16","med8","small8"],default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--exclude",help='exclude feature from training',default=None)
	#parser.add_argument("--reweightClasses",help="reweight classes",default=False,action='store_true')
	parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=False,action='store_true')
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
