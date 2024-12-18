import argparse
from ProcessData import CSVReader
import pandas as pd
from ConvNN import ConvNeuralNetwork
from dualConvNN import dualConvNeuralNetwork

# DNN for identifying detector background (spikes + beam halo) from physics bkg
def runDNN(args):
	#using AL1IsoPho presel s.t. there is no MET cut to bias the presence + spectrum of detector bkgs in MET PD
	#AL1IsoPho = at least 1 isolated photon (standard presel iso)
	#data
	reader = CSVReader()
	reader.AddFile("csv/MET_R17_AL1IsoPho_v22_MET_AOD_Run2017E_17Nov2017_superclusters_defaultv4.csv")
	#reader.AddFile("csv/DEG_R17_AL1IsoPho_v22_DoubleEG_AOD_Run2017F_09Aug2019_UL2017_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/JetHT_R17_AL1IsoPho_v22_JetHT_AOD_Run2017F_17Nov2017_superclusters_defaultv3p5.csv")
	
	#METreader = CSVReader()
	#METreader.AddFile("csv/MET_R17_AL1IsoPho_v22_MET_AOD_Run2017E_17Nov2017_superclusters_defaultv3p5.csv")
	#DEGreader = CSVReader()
	#DEGreader.AddFile("csv/DEG_R17_AL1IsoPho_v22_DoubleEG_AOD_Run2017F_09Aug2019_UL2017_superclusters_defaultv3p5.csv")
	#JetHTreader = CSVReader()
	#JetHTreader.AddFile("csv/JetHT_R17_AL1IsoPho_v22_JetHT_AOD_Run2017F_17Nov2017_superclusters_defaultv3p5.csv")

	#MC background
	#reader.AddFile("csv/GJets_R17_AL1IsoPho_v22_GJets_HT-40To100_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/GJets_R17_AL1IsoPho_v22_GJets_HT-40To100_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/GJets_R17_AL1IsoPho_v22_GJets_HT-100To200_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/GJets_R17_AL1IsoPho_v22_GJets_HT-200To400_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/GJets_R17_AL1IsoPho_v22_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/GJets_R17_AL1IsoPho_v22_GJets_HT-600ToInf_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	##
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT50to100_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT100to200_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT200to300_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT300to500_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT500to700_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT700to1000_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT1000to1500_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT1500to2000_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/QCD_R17_AL1IsoPho_v22_QCD_HT2000toInf_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p5.csv")
	##
	
	
	reader.CleanData()
	data = reader.GetData()

	catToName = {1 : "physicsBkg", 2 : "beamHalo", 3 : "spike", 0 : "signal"}
	catToColor = {1 : "green", 2 : "red", 3 : "orange", 0 : "pink"}
	tot = len(data)
	phys = len(data[data["label"] == 1])
	BH = len(data[data["label"] == 2])
	spike = len(data[data["label"] == 3])
	print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
	#need to viz inputs before transforming labels bc relies on integer labels

	#balance classes via random undersampling - default
	print("Even out classes")
	sizes = {}
	sizes[len(data[data["label"] == 1])] = 1
	sizes[len(data[data["label"] == 2])] = 2
	sizes[len(data[data["label"] == 3])] = 3

	print(sizes)
	nsamp = min(sizes.keys())
	lab = sizes[nsamp] #label of min sample
	
	data_samples = []	
	for l in sizes.values():
		if l == lab:
			continue	
		data_samples.append(data.query('label == '+str(l)).sample(n=nsamp,random_state=111))
	data_samples.append(data.query('label == '+str(lab)))
	data = pd.concat(data_samples)
	tot = len(data)
	phys = len(data[data["label"] == 1])
	BH = len(data[data["label"] == 2])
	spike = len(data[data["label"] == 3])
	print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
	
	if(args.dryRun):
		exit()


		
	#"features" to use for training
	#for a CNN this is just a weighted map of the subclusters in eta-phi 2D space	
	
	network_name = "KU-CNN_detector"
	if args.extra is not None:
		network_name += "_"+args.extra
	nepochs = int(args.nEpochs)
	early = False
	channels = []
	if(args.cols == "default"):
		#default input set
		channels = ["E","t","r"]
	elif(args.cols == "Eonly"):
		channels = ["E"]
		network_name += "_"+args.cols
	elif(args.cols == "timeOnly"):
		channels = ["t"]
		network_name += "_"+args.cols
	elif(args.cols == "rOnly"):
		channels = ["r"]
		network_name += "_"+args.cols
	elif(args.cols == "ErOnly"):
		channels = ["E","r"]
		network_name += "_"+args.cols
	elif(args.cols == "EMultr"):
		channels = ["E","r"]
		network_name += "_"+args.cols
	elif(args.cols == "normE"):
		channels = ["E"]
		network_name += "_"+args.cols
	elif(args.cols == "normEMultr"):
		channels = ["E","r"]
		network_name += "_"+args.cols
	else:
		print("Invalid features selected",args.network)
		exit()

	network_name += "_"+str(nepochs)+"epochs"
	if(early):
		network_name += "_earlyStop"
	
	
	#len(filters) = # layers
	#filters[i] = # filters at ith layer
	if(args.arch == "default"):
		network_name += "_"+args.arch
		filters = [64, 64, 64] 
		model = ConvNeuralNetwork(data,filters,network_name,channels)
		model.BuildModel()
	elif(args.arch == "small8"):
		network_name += "_"+args.arch
		filters = [8, 8, 8] 
		model = ConvNeuralNetwork(data,filters,network_name,channels)
		model.BuildModel()
	elif(args.arch == "dual"):
		network_name += "_"+args.arch
		filters = [8, 8, 8] 
		model = dualConvNeuralNetwork(data,filters,network_name,channels)
		mask1 = (3,3) #spikes
		mask2 = (3,1) #beam halo
		model.BuildModel(mask1, mask2)
	else:
		print("Invalid architecture selected",args.network)
		exit()
	
	
	model.SetCategoryNames(catToName,catToColor)
	#visualize inputs
	model.VizInputs()
	model.CompileModel()
	model.summary()
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	model.TestModel(1,True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch','-a',help="which architecture to run",choices=["default","small8","dual"],default="default")
	parser.add_argument('--cols','-c',help="which set of inputs to run",choices=["default","Eonly","timeOnly","rOnly","ErOnly","EMultr","normE","normEMultr"],default="default")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
