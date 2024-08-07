from ProcessData import CSVReader
from DeepNN import DeepNeuralNetwork


#def main():
reader = CSVReader("csv/photonSkim_test_emAlpha0p500_thresh1p000_NperGeV0p100_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_output257_v16.csv")
reader.CleanData()
data = reader.GetData()
#GJets csv has all unmatched labels

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
