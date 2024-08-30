from ProcessData import CSVReader
#from DeepNN import DeepNeuralNetwork

#locally made
#reader = CSVReader("csv/superclusterSkimNperGeV0p033_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_output446_v20.csv")
#reader.AddFile("csv/superclusterSkimNperGeV0p033_GMSB_L-350TeV_Ctau-200cm_AODSIM_RunIIFall17DRPremix_output18_v20.csv")
#reader.AddFile("csv/superclusterSkimNperGeV0p033_MET_AOD_Run2017E_17Nov2017_output108_v20.csv")
#condor files
reader = CSVReader("csv/MET_R17E_MET75_v20_MET_AOD_Run2017E_17Nov2017_superclusters_defaultv3p4_noBHFilter.csv")
reader.AddFile("csv/GJets_R17_MET75_v20_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_superclusters_defaultv3p4_noBHFilter.csv")
reader.CleanData()
data = reader.GetData()
tot = len(data)
sig = len(data[data["label"] == 0])
nom = len(data[data["label"] == 1])
BH = len(data[data["label"] == 2])
print(tot, "subclusters, {:.2f}% sig {:.2f}% nom {:.2f}% BH".format(sig/tot,nom/tot,BH/tot))
#EVEN OUT CLASSES - REMOVE NOM RANDOMLY TO MATCH ~2k BH SUBCLUSTER


'''
#len(nodes) = # layers
#nodes[i] = # nodes at ith layer
nodes = [64, 64, 64] #simple-DNN

model = DeepNeuralNetwork(data,nodes)
model.BuildModel("simple-DNN")
model.CompileModel()
model.summary()
#input is TrainModel(epochs=1,oname="",int:verb=1)
#caltech: epochs = 50, batch size = 10k, early stopping, adaptive learning rate
model.TrainModel(10,True)
#needs test data + to make ROC plots
model.TestModel(1,True)
'''
