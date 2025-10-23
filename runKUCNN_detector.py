import argparse
from ProcessData import CSVReader
import pandas as pd
from ConvNN import ConvNeuralNetwork
from dualConvNN import dualConvNeuralNetwork
import numpy as np

# DNN for identifying detector background (spikes + beam halo) from physics bkg
def runDNN(args):
    #using AL1IsoPho presel s.t. there is no MET cut to bias the presence + spectrum of detector bkgs in MET PD
    #AL1IsoPho = at least 1 isolated photon (standard presel iso)
    #data
    printstats = True
    reader = CSVReader(printstats)
    reader.AddFile("csv/MET_R17_AL1NpSC_nolumimask_v31_MET_AOD_Run2017B-09Aug2019_UL2017_rsb-v1_superclusters_defaultv7_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")
    reader.AddFile("csv/DoubleEG_R17_AL1SelEle_nolumimask_v31_DoubleEG_AOD_Run2017B-09Aug2019_UL2017-v1_superclusters_defaultv7_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")
    reader.AddFile("csv/MET_R18_AL1NpSC_DEOnly_v31_MET_AOD_Run2018B-15Feb2022_UL2018-v1_superclusters_defaultv7_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")
    reader.AddFile("csv/EGamma_R18_AL1SelEle_nolumimask_v31_EGamma_AOD_Run2018C-15Feb2022_UL2018-v1_superclusters_defaultv7_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")

    reader.CleanData()
    reader.SelectClass(1,["EGamma","DoubleEG"]); #choose for a certain class (first arg) to only come from sample (second arg)
    #reader.SelectClass(1,"DoubleEG"); #choose for a certain class (first arg) to only come from sample (second arg)


    #set max number of samples with label to be nsamp
    #reader.CapClass(3,3000)
    
    #balance classes via random undersampling - default
    reader.BalanceClasses([1,2,3])
    data = reader.GetData()
    
    catToName = {1 : "physicsBkg", 2 : "beamHalo", 3 : "spike", 0 : "signal"}
    catToColor = {1 : "green", 2 : "red", 3 : "orange", 0 : "pink"}
    
    
    	
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
 
    if args.arch not in arch_map.keys():
    	print("Invalid architecture selected",args.network)
    	exit()
    
    filters = arch_map[args.arch] 
    
    model = ConvNeuralNetwork(data,filters,network_name)
    model.BuildModel()
    model.SetCategoryNames(catToName,catToColor)
    if(args.testNetwork):
        print("Evaluating network",network_name)
        model.TestModel(1,True)
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
    model.TestModel(1,True)
    model.VizModelWeights()
    model.VizFeatureMaps()
    
    #test on JetHT
    #jetHT_reader = CSVReader(printstats)
    #model.TestModel_data(MC_reader.GetData(), True,"GJets_HT400to600")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch','-a',help="which architecture to run",choices=["default","small8","small4","small3","small2","xsmall3"],default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument('--testNetwork',help='evaluate trained network specified by other flags',default=False,action='store_true')
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
