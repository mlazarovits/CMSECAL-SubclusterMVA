import argparse
from ProcessData import CSVReader
import pandas as pd
from DeepNN import DeepNeuralNetwork


def runDNN(args):
	#condor files
	#signal
	reader = CSVReader("csv/GMSB_R17_MET100_v21_GMSB_L-250TeV_Ctau-10cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-300TeV_Ctau-400cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-300TeV_Ctau-600cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-300TeV_Ctau-1000cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-350TeV_Ctau-0_1cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-350TeV_Ctau-10cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-350TeV_Ctau-200cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-350TeV_Ctau-800cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-400TeV_Ctau-800cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GMSB_R17_MET100_v21_GMSB_L-400TeV_Ctau-200cm_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	##MC background
	#
	reader.AddFile("csv/GJets_R17_MET100_v21_GJets_HT-40To100_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GJets_R17_MET100_v21_GJets_HT-100To200_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GJets_R17_MET100_v21_GJets_HT-200To400_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GJets_R17_MET100_v21_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/GJets_R17_MET100_v21_GJets_HT-600ToInf_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	#
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT50to100_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT100to200_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT200to300_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT300to500_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT500to700_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT700to1000_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT1000to1500_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT1500to2000_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	reader.AddFile("csv/QCD_R17_MET100_v21_QCD_HT2000toInf_AODSIM_RunIIFall17DRPremix_photons_defaultv3p5.csv")
	#
	#data
	#reader = CSVReader("csv/MET_R17E_MET100_v21_MET_AOD_Run2017E_17Nov2017_photons_defaultv3p5.csv")
	#reader.AddFile("csv/DEG_R17_MET100_v21_DoubleEG_AOD_Run2017F_09Aug2019_UL2017_photons_defaultv3p5.csv")
	
	reader.CleanData()
	data = reader.GetData()
	
	tot = len(data)
	sig = len(data[data["label"] == 1])
	nom = len(data[data["label"] == 0])
	BH = len(data[data["label"] == 2])
	print(" ",tot, ("subclusters, sig: "+str(sig)+" {:.2f}%, nom: "+str(nom)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(sig/tot,nom/tot,BH/tot))


	
	#select BH to be nom (not distinguished in caltech NN)
	print("Set BH to nom")
	data["label"] = data["label"].mask(data["label"] == 2,1)
	tot = len(data)
	sig = len(data[data["label"] == 1])
	nom = len(data[data["label"] == 0])
	BH = len(data[data["label"] == 2])
	d = len(data[data["sample"] == "METPD"]) + len(data[data["sample"] == "EGamma"]) 
	gjets = len(data[data["sample"].str.contains("GJets")])
	qcd = len(data[data["sample"].str.contains("QCD")])
	mc = gjets + qcd
	gmsb = len(data[data["sample"].str.contains("GMSB")])
	print(" ",tot, ("subclusters, sig: "+str(sig)+" {:.2f}%, nom: "+str(nom)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(sig/tot,nom/tot,BH/tot))
	print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, GMSB "+str(gmsb)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	
	
	
	
	#make sure all SCs are matched to photon (no -999s)
	print("Remove SCs that did not pass Caltech object selection")
	data = data[data.R9 != -999]
	#print((data == -999).any().any())
	tot = len(data)
	sig = len(data[data["label"] == 1])
	nom = len(data[data["label"] == 0])
	BH = len(data[data["label"] == 2])
	d = len(data[data["sample"] == "METPD"]) + len(data[data["sample"] == "EGamma"]) 
	gjets = len(data[data["sample"].str.contains("GJets")])
	qcd = len(data[data["sample"].str.contains("QCD")])
	mc = gjets + qcd
	gmsb = len(data[data["sample"].str.contains("GMSB")])
	#print(tot, ("subclusters, "+str(d)+" {:.2f}% data "+str(gjets)+" {:.2f}% GJets "+str(qcd)+" {:.2f}% QCD "+str(gmsb)+" {:.2f}% gmsb").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	print(" ",tot, ("subclusters, sig "+str(sig)+" {:.2f}%, nom "+str(nom)+" {:.2f}%, BH "+str(BH)+" {:.2f}%").format(sig/tot,nom/tot,BH/tot))
	print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, GMSB "+str(gmsb)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))



	#only take leading photons
	print("Select leading photons")
	data = data[data.lead == 1]
	tot = len(data)
	sig = len(data[data["label"] == 1])
	nom = len(data[data["label"] == 0])
	BH = len(data[data["label"] == 2])
	d = len(data[data["sample"] == "METPD"]) + len(data[data["sample"] == "EGamma"]) 
	gjets = len(data[data["sample"].str.contains("GJets")])
	qcd = len(data[data["sample"].str.contains("QCD")])
	mc = gjets + qcd
	gmsb = len(data[data["sample"].str.contains("GMSB")])
	#print(tot, ("subclusters, "+str(d)+" {:.2f}% data "+str(gjets)+" {:.2f}% GJets "+str(qcd)+" {:.2f}% QCD "+str(gmsb)+" {:.2f}% gmsb").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	print(" ",tot, ("subclusters, sig "+str(sig)+" {:.2f}%, nom "+str(nom)+" {:.2f}%, BH "+str(BH)+" {:.2f}%").format(sig/tot,nom/tot,BH/tot))
	print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, GMSB "+str(gmsb)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	

	'''
	#randomly select # sig == # nom entries
	if sig < nom:
		lab = 1
	else:
		lab = 0
	print("Even out classes")
	datanom = data.query('label == '+str(lab)).sample(n=min(sig,nom),random_state=111)
	data = data.query('label != '+str(lab))
	data = pd.concat([data,datanom])
	tot = len(data)
	sig = len(data[data["label"] == 0])
	nom = len(data[data["label"] == 1])
	BH = len(data[data["label"] == 2])
	d = len(data[data["sample"] == "METPD"]) + len(data[data["sample"] == "EGamma"]) 
	gjets = len(data[data["sample"].str.contains("GJets")])
	qcd = len(data[data["sample"].str.contains("QCD")])
	mc = gjets + qcd
	gmsb = len(data[data["sample"].str.contains("GMSB")])
	#print(tot, ("subclusters, "+str(sig)+" {:.2f}% sig "+str(nom)+" {:.2f}% nom "+str(BH)+" {:.2f}% BH").format(sig/tot,nom/tot,BH/tot))
	print(" ",tot, ("subclusters, sig "+str(sig)+" {:.2f}%, nom "+str(nom)+" {:.2f}%, BH "+str(BH)+" {:.2f}%").format(sig/tot,nom/tot,BH/tot))
	print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, GMSB "+str(gmsb)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,gmsb/tot))
	print(data["label"].unique())
	'''
	if(args.dryRun):
		exit()
	
	network_name = "KU-DNN"
	if args.extra is not None:
		network_name += "_"+args.extra
	nepochs = 2000
	early = False
	if(args.network == "default"):
		#default input set
		cols = ["sample","event","object","subcl","eta_sig","phi_sig","etaphi_cov","timeeta_cov","label"]
	else:
		print("Invalid network selected",args.network)

	print("features used",cols)	
	data = data[cols]
	network_name += "_"+str(nepochs)+"epochs"
	if(early):
		network_name += "_earlyStop"
	
	
	#len(nodes) = # layers
	#nodes[i] = # nodes at ith layer
	nodes = [64, 64, 64] #simple-DNN
	
	
	model = DeepNeuralNetwork(data,nodes,network_name)
	model.BuildModel()
	#visualize inputs
	model.CompileModel()
	model.summary()
	#input is TrainModel(epochs=1,oname="",int:verb=1)
	#caltech: epochs = 500, batch size = 10k, early stopping, adaptive learning rate
	model.TrainModel(nepochs,batch=100,viz=True,savebest=True,earlystop=early)
	#needs test data + to make ROC plots
	model.TestModel(1,True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--network','-n',help="which set of inputs to run",required=True,choices=["default"])
	parser.add_argument("--dryRun",help="dry run - stats only (don't run network)",action='store_true')
	parser.add_argument("--extra",'-e',help='extra string for network name')
	args = parser.parse_args()

	runDNN(args)

if __name__ == "__main__":
	main()
