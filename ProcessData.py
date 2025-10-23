import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#add multiple files
class CSVReader:
    def __init__(self, file, printStats = False):
        #data is a list of feature, label pairs or tuples
        self._data = np.array([])
        self._file = file
        self._header = np.array([])
        self._data = pd.read_csv(self._file)
        #dictionary for integer labels to strings for plotting
        self._labelsDict = {0 : "sig", 1 : "physics", 2 : "BH", 3 : "spike"}
        self._printstats = printStats
    
    def __init__(self, printStats = False):
        #data is a list of feature, label pairs or tuples
        self._data = np.array([])
        self._header = np.array([])
        self._data = pd.DataFrame()
        #dictionary for integer labels to strings for plotting
        self._labelsDict = {0 : "sig", 1 : "physics", 2 : "BH", 3 : "spike"}
        self._printstats = printStats

    def AddFile(self,file):
        data = pd.read_csv(file)
        self._data = pd.concat([self._data,data],ignore_index=True)

    #as files are added, drop rows from given samp that are not nclass = [2,3]
    #def AddDetBkgSource(self, file):

    #as files are added, drop rows from given samp that are not nclass = 1
    #def AddPhysBkgSource(self, file):

    #as files are added, drop rows from given samp that are not nclass = 1, up to nrows (randomly selected)
    #def AddPhysBkgSourceLimited(self, file, nrows)


    def AddLargeFile(self,file,nsamp=1000,chunksize=1e5):
        if nsamp > chunksize:
            print("nsamp",nsamp,"cannot be larger than chunksize",chunksize)
            return
        largedata = []
        if not self._data.empty:
            largedata = largedata.append(self._data)
        #total # of samples will be floor(nrows/chunksize)*nsamp
        for nchunk, chunk in enumerate(pd.read_csv(file,chunksize=chunksize)):
            if(len(largedata) > 10 and nsamp != -1):
                break
            print("processing chunk #",nchunk)
            if(nsamp != -1):
                chunk = chunk.sample(n=nsamp,random_state=111)
            largedata.append(chunk)
        self._data = pd.concat(largedata,ignore_index=True)
        print("finishing concating dfs - have total of",len(self._data),"rows")
    def PrintStats(self):
        sig = len(self._data[self._data["label"] == 0])
        nom = len(self._data[self._data["label"] == 1])
        #gmsb = len(self._data[self._data["sample"].str.contains("GMSB") == True])
        gogo = len(self._data[self._data["sample"].str.contains("gogo") == True])
        sqsq = len(self._data[self._data["sample"].str.contains("sqsq") == True])
        if gogo == 0:
        	sig = sqsq
        gjets = len(self._data[self._data["sample"].str.contains("GJets") == True])
        qcd = len(self._data[self._data["sample"].str.contains("QCD") == True])
        datasamples = ["METPD","EGamma","DoubleEG","JetHT"]
        d = 0
        for i in datasamples:
        	d += len(self._data[self._data["sample"] == i])
        tot = len(self._data)
        phys = len(self._data[self._data["label"] == 1])
        BH = len(self._data[self._data["label"] == 2])
        spike = len(self._data[self._data["label"] == 3])
        print(" ",tot, ("subclusters, phys: "+str(phys)+" {:.2f}%, spike: "+str(spike)+" {:.2f}%, BH: "+str(BH)+" {:.2f}%").format(phys/tot,spike/tot,BH/tot))
        print(" ",tot, ("subclusters, data: "+str(d)+" {:.2f}%, GJets: "+str(gjets)+" {:.2f}%, QCD "+str(qcd)+" {:.2f}%, signal "+str(sig)+" {:.2f}%").format(d/tot,gjets/tot,qcd/tot,sig/tot))
    
        
    #data cleaning, cuts, etc.
    #unmatched = -1
    #signal = 0
    #!signal = 1 
    #BH = 2
    #spike = 3
    def CleanData(self):
        print("Cleaning data",len(self._data),"subclusters initially")
        #remove any "unmatched" labels
        self._data = self._data[self._data['label'] != -1]
        if(self._printstats):
            print("after unmatched removal")
            self.PrintStats()
    
        #remove not-signal-matched photons in GMSB sample (this is not the "bkg" we want to target)
        rowbool = ((self._data["sample"].str.contains("GMSB") == True) & (self._data["label"] == 1)) 
        self._data = self._data.drop(self._data[rowbool].index)  
        if(self._printstats):
            print("after GMSB bkg removal")
            self.PrintStats()

        #put extra cuts on subcluster energy, etc.  
        self._data.dropna(how="any")
        if(self._printstats):
            print('after dropna')
            self.PrintStats()

    #only use SCs with 1 subcluster
    #can remove if CSVs are updated accordingly
    def LeadingOnly(self):
        #if entry in (evt, object) appears multiple times, skip all rows with (evt, obj)
        maxevtnum = max(self._data["event"])+1
        for e in range(maxevtnum):
            if len(self._data[self._data["event"] == e]) < 1:
                continue
            maxobjnum = max(self._data[self._data["event"] == e]["object"])+1
            for o in range(maxobjnum):
                rowbool = ((self._data["event"] == e) & (self._data["object"] == o))
                nEntries = len(self._data[rowbool])
                if nEntries > 1:
                    #print("event",e,"obj",o,"has",nEntries,"subcls with max energy",maxenergy)
                    #drop all rows with this event and object
                    self._data = self._data.drop(self._data[rowbool].index)
        if(self._printstats):
            print('after dropping SCs with multiple subclusters')
            self.PrintStats()

    #remove subleading subclusters
    def RemoveSubleading(self):
        #save + return dataframe with rows of all subleading subclusters
        sublead_data = self._data[self._data['subcl'] != 0]
        self._data = self._data[self._data['subcl'] == 0]
        if(self._printstats):
            print('after removing subleading subclusters')
            self.PrintStats()
        return sublead_data


    def SelectClass(self,nclass,samps):
        #drop rows from all samps that are not nclass
        self._data = self._data[~((self._data["sample"].isin(samps)) & (self._data["label"] != nclass))]
        ##drop rows for !samp that are nclass
        self._data = self._data[~((self._data["label"] == nclass) & (~self._data["sample"].isin(samps)))]	
        if(self._printstats):
            print("after setting class",nclass,"to be only from",samps)
            self.PrintStats()

    #only allow nsamp of samples from nclass
    def CapClass(self,nclass,nsamp):
        sampled_subset = self._data[self._data['label'] == nclass].sample(n=nsamp, random_state=42)	
        #replace all rows with label l by this sampled subset
        self._data = pd.concat([self._data[self._data['label'] != nclass], sampled_subset], ignore_index=True)


    def BalanceClasses(self, labels):
        sizes = {}
        for l in labels:
            sizes[len(self._data[self._data["label"] == l])] = l
        nsamp = min(sizes.keys())
        lab = sizes[nsamp] #label of min sample
        
        data_samples = []   
        for l in sizes.values():
            if l == lab:
                continue    
            data_samples.append(self._data.query('label == '+str(l)).sample(n=nsamp,random_state=111))
        data_samples.append(self._data.query('label == '+str(lab)))
        self._data = pd.concat(data_samples)
        if(self._printstats):
            print('after balancing classes')
            self.PrintStats()

    def GetData(self):
        return self._data

