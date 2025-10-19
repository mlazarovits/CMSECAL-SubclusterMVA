import argparse
from ProcessData import CSVReader
import pandas as pd
from DeepNN import DeepNeuralNetwork

# DNN for bkg classification (iso vs noniso) 
def runDNN(args):
	#data
	reader = CSVReader()
	#reader.AddFile("csv/MET_R17_AL1IsoPho_v22_MET_AOD_Run2017E_17Nov2017_superclusters_defaultv4.csv")
	#reader.AddFile("csv/DEG_R17_AL1IsoPho_v22_DoubleEG_AOD_Run2017F_09Aug2019_UL2017_superclusters_defaultv3p5.csv")
	#reader.AddFile("csv/JetHT_R17_AL1IsoPho_v22_JetHT_AOD_Run2017F_17Nov2017_superclusters_defaultv3p5.csv")
	reader.AddFile("csv/GJets_R18_InvMetPho30_v31_GJets_HT-400To600_TuneCP5_AODSIM_RunIISummer20UL18RECO_photons_defaultv3p10_noIso_isoBkgSel_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")
	reader.AddFile("csv/QCD_R18_InvMET100_v31_QCD_HT300to500_TuneCP5_AODSIM_RunIISummer20UL18RECO_photons_defaultv3p10_noIso_beta0-1e-5_m0-0p0-0p0-0p0_W0diag-0p013-0p013-33p333_nu0-3_NperGeV-0p0333333_emAlpha-1e-5.csv")	
	
	
	reader.CleanData()
	data = reader.GetData()

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
	tot = len(data)
	isobkg = len(data[data["label"] == 4])
	nonisobkg = len(data[data["label"] == 6])
	print(" ",tot, ("subclusters, iso bkg: "+str(isobkg)+" {:.2f}%, noniso bkg: "+str(nonisobkg)+" {:.2f}%").format(isobkg/tot,nonisobkg/tot))
	#need to viz inputs before transforming labels bc relies on integer labels

	'''
	#by process breakdown
	print("\nMET sample")
	METreader.CleanData()
	data = METreader.GetData()
	tot = len(data)
	phys = len(data[data["label"] == 1])
	BH = len(data[data["label"] == 2])
	spike = len(data[data["label"] == 3])
	print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
	
	print("DoubleEG sample")
	DEGreader.CleanData()
	data = DEGreader.GetData()
	tot = len(data)
	phys = len(data[data["label"] == 1])
	BH = len(data[data["label"] == 2])
	spike = len(data[data["label"] == 3])
	print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
	
	print("JetHT sample")
	JetHTreader.CleanData()
	data = JetHTreader.GetData()
	tot = len(data)
	phys = len(data[data["label"] == 1])
	BH = len(data[data["label"] == 2])
	spike = len(data[data["label"] == 3])
	print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
	'''

	#randomly select # sig == # nom entries
	print("Even out classes")
	sizes = {}
	sizes[len(data[data["label"] == 4])] = 4
	sizes[len(data[data["label"] == 6])] = 6

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
	isobkg = len(data[data["label"] == 4])
	nonisobkg = len(data[data["label"] == 6])
	print(" ",tot, ("subclusters, iso bkg: "+str(isobkg)+" {:.2f}%, noniso bkg: "+str(nonisobkg)+" {:.2f}%").format(isobkg/tot,nonisobkg/tot))
	


		
	
	network_name = "KU-DNN_photonID"
	if args.extra is not None:
		network_name += "_"+args.extra
	nepochs = int(args.nEpochs)
	early = False
	shape_cols = ["sample","event","object","subcl","eta_sig","phi_sig","etaphi_cov","major_length", "minor_length","label"]
	iso_cols = ["sample","event","object","subcl","trkSumPtSolidConeDR04","hadTowOverEM","ecalRHSumEtConeDR04","label"]
	if(args.network == "shape"):
		#default input set
		cols = shape_cols 
	elif(args.network == "iso"):
		#default input set
		cols = iso_cols 
	elif(args.network == "isoShape"):
		cols = shape_cols + iso_cols
		cols = set(cols)
		cols = list(cols) 
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
	data = data[cols]
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
	
	
	model = DeepNeuralNetwork(data,nodes,network_name)
	model.SetCategoryNames(catToName,catToColor)
	model.VizInputs()
	model.BuildModel()
	#visualize inputs
	model.CompileModel()
	model.summary()
	if(args.dryRun):
		exit()
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	model.TestModel(1,True,validate_model=True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--network','-n',help="which set of inputs to run",choices=["iso","shape","isoShape"],required=True)
	parser.add_argument('--arch','-a',help="which architecture to run",choices=["default","med16","med8","small8"],default="small3")
	parser.add_argument('--nEpochs',help="number of epochs for training",default=20)
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true',default=False)
	parser.add_argument("--extra",'-e',help='extra string for network name')
	parser.add_argument("--exclude",help='exclude feature from training',default=None)
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
