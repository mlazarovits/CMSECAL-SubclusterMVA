from ProcessData import CSVReader
import pandas as pd
from DeepNN import DeepNeuralNetwork

#locally made
#reader = CSVReader("csv/superclusterSkimNperGeV0p033_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_output446_v20.csv")
#reader.AddFile("csv/superclusterSkimNperGeV0p033_GMSB_L-350TeV_Ctau-200cm_AODSIM_RunIIFall17DRPremix_output18_v20.csv")
#reader.AddFile("csv/superclusterSkimNperGeV0p033_MET_AOD_Run2017E_17Nov2017_output108_v20.csv")

#condor files
#signal
reader = CSVReader("csv/GMSB_R17_MET75_v20_GMSB_L-250TeV_Ctau-10cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-300TeV_Ctau-400cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-300TeV_Ctau-600cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-300TeV_Ctau-1000cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-350TeV_Ctau-0_1cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-350TeV_Ctau-10cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-350TeV_Ctau-200cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-350TeV_Ctau-800cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-400TeV_Ctau-800cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GMSB_R17_MET75_v20_GMSB_L-400TeV_Ctau-200cm_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
#MC background
reader.AddFile("csv/GJets_R17_MET75_v20_GJets_HT-40To100_AODSIM_RunIIFall17DRPremix01_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GJets_R17_MET75_v20_GJets_HT-100To200_AODSIM_RunIIFall17DRPremix1_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GJets_R17_MET75_v20_GJets_HT-200To400_AODSIM_RunIIFall17DRPremix1_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GJets_R17_MET75_v20_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GJets_R17_MET75_v20_GJets_HT-600ToInf_AODSIM_RunIIFall17DRPremix3_superclusters_defaultv3p4_noBHFilter.csv")

reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT50to100_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT100to200_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT200to300_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT300to500_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT500to700_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT700to1000_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT1000to1500_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT1500to2000_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/QCD_R17_MET75_v20_QCD_HT2000toInf_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")

#data
#reader = CSVReader("csv/MET_R17E_MET75_v20_MET_AOD_Run2017E_17Nov2017_superclusters_defaultv3p4_noBHFilter.csv")
#reader.AddFile("csv/DEG_R17_MET75_v20_DoubleEG_AOD_Run2017F_09Aug2019_UL2017_superclusters_defaultv3p4_noBHFilter.csv")

reader.CleanData()
data = reader.GetData()
tot = len(data)
sig = len(data[data["label"] == 0])
nom = len(data[data["label"] == 1])
BH = len(data[data["label"] == 2])
print(tot, ("subclusters, "+str(sig)+" {:.2f}% sig "+str(nom)+" {:.2f}% nom "+str(BH)+" {:.2f}% BH").format(sig/tot,nom/tot,BH/tot))

#select BH to be nom (not distinguished in caltech NN)
print("Set BH to nom")
data["label"] = data["label"].mask(data["label"] == 2,1)
tot = len(data)
sig = len(data[data["label"] == 0])
nom = len(data[data["label"] == 1])
BH = len(data[data["label"] == 2])
print(tot, ("subclusters, "+str(sig)+" {:.2f}% sig "+str(nom)+" {:.2f}% nom "+str(BH)+" {:.2f}% BH").format(sig/tot,nom/tot,BH/tot))

#select benchmark features to train on - R9, S_ieie, smaj, smin
benchmark = ["Sample","Event","supercl","subcl","R9","Sietaieta","Smajor","Sminor","ecalPFClusterIsoOvPt","hcalPFClusterIsoOvPt","trkSumPtHollowConeDR03OvPt","label"]


data = data[benchmark]
#make sure all SCs are matched to photon (no -999s)
print("Remove SCs that did not pass Caltech object selection")
data = data[data.R9 != -999]
#print((data == -999).any().any())
tot = len(data)
sig = len(data[data["label"] == 0])
nom = len(data[data["label"] == 1])
print(tot, ("subclusters, "+str(sig)+" {:.2f}% sig "+str(nom)+" {:.2f}% nom ").format(sig/tot,nom/tot))
d = len(data[data["Sample"] == "METPD"]) + len(data[data["Sample"] == "notFound"]) #change to DEG when fixed
mc = len(data[data["Sample"].str.contains("GJets")])
gmsb = len(data[data["Sample"].str.contains("GMSB")])
print(tot, ("subclusters, "+str(d)+" {:.2f}% data "+str(mc)+" {:.2f}% mc bkg "+str(gmsb)+" {:.2f}% gmsb").format(d/tot,mc/tot,gmsb/tot))

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
print(tot, ("subclusters, "+str(sig)+" {:.2f}% sig "+str(nom)+" {:.2f}% nom "+str(BH)+" {:.2f}% BH").format(sig/tot,nom/tot,BH/tot))
print(data["label"].unique())

#len(nodes) = # layers
#nodes[i] = # nodes at ith layer
nodes = [64, 64, 64] #simple-DNN

model = DeepNeuralNetwork(data,nodes)
model.BuildModel("Caltech-DNN")
#visualize inputs
model.VizInputs()
model.CompileModel()
model.summary()
#input is TrainModel(epochs=1,oname="",int:verb=1)
#caltech: epochs = 500, batch size = 10k, early stopping, adaptive learning rate
model.TrainModel(500,viz=True,savebest=True)
#needs test data + to make ROC plots
model.TestModel(1,True)

