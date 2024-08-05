from ProcessData import CSVReader
from DeepNN import DeepNeuralNetwork


#def main():
reader = CSVReader("csv/photonSkim_test_emAlpha0p500_thresh1p000_NperGeV0p100_GJets_HT-400To600_AODSIM_RunIIFall17DRPremix_output257_v16.csv")
reader.CleanData()
data = reader.GetData()

print(data,data.shape)
#len(nodes) = # layers
#nodes[i] = # nodes at ith layer
nodes = [64, 64, 64]
datashape = data.shape

model = DeepNeuralNetwork(datashape[1],nodes)
model.BuildModel()
model.CompileModel()
model.summary()
#model.SetData(data) - cleans out unnecessary columsn (ie subcl #) and sets train/test split
#model.TrainModel()
#needs test data + to make ROC plots
#model.Evaluate(test_data, batch)
