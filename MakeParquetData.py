import numpy as np
import argparse
from ConvertData import TTreeReader

def make_glgl_samples(reader):
	#("root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-1500_mN2-500_mN1-100-ct0p1_superclusters_defaultv9p2.root","",step_size=10000,maxnchunk=-1,debug=False,labelas=1)
	glglz = ["root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-1500_mN2-500_mN1-100-ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-200-ct0p001_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-200-ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-200-ct0p3_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-350-ct0p001_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-350-ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-350-ct0p3_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1950_mN1-1900-ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-200-ct0p001_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-200-ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-200-ct0p3_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-350-ct0p001_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-350-ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-350-ct0p3_superclusters_defaultv9p2.root"]

	glgl = ["root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-1500_mN2-500_mN1-100_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-250_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-500_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-1000_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-250_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-500_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1000_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1500_superclusters_defaultv9p2.root", "root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-250_superclusters_defaultv9p2.root", "root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-500_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1000_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1500_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1900_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-250_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-500_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-500_mN1-1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-500_mN1-250_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2500_mN2-1500_mN1-1000_superclusters_defaultv9p2.root"]
	
	
	sqsq = ["root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1700_mN2-1500_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1700_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1850_mN2-1650_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1850_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2000_mN2-1800_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2000_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2150_mN2-1950_mN1-100_ct0p1_superclusters_defaultv9p2.root","root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2150_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root"]
	return glgl, sqsq, glglz

def main(args):
	reader_train = TTreeReader(args.obj, args.objType)
	reader_test = TTreeReader(args.obj, args.objType,"test")
	step_size = 10000
	'''
	if args.obj == "SC":
		max_egamma_chunk = 2000
		#all already on LPC
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_superclusters_defaultv9p1_EGamma_R18_InvMetPho30_NoSV_v31_EGamma_AOD_Run2018C.root","EGamma18_RunC",step_size,max_egamma_chunk)
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_superclusters_defaultv9p1_DoubleEG_R17_InvMetPho30_v31_DoubleEG_AOD_Run2017B-09Aug2019_UL2017-v1.root","EGamma17_RunB",step_size,max_egamma_chunk)
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_superclusters_defaultv9p1_MET_R18_AL1NpSC_DEOnly_v31_MET_RunB_2018.root","METPD18_RunB",step_size)
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_superclusters_defaultv9p1_MET_R18_AL1NpSC_DEOnly_v31_MET_AOD_Run2018A-15Feb2022_UL2018-v1.root","METPD18_RunA",step_size)
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_superclusters_defaultv9p1_MET_R17_AL1NpSC_nolumimask_v31_MET_AOD_Run2017B-09Aug2019_UL2017_rsb-v1.root","METPD17_RunB",step_size)
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_superclusters_defaultv9p1_MET_R17_AL1NpSC_nolumimask_v31_MET_AOD_Run2017D-09Aug2019_UL2017_rsb-v1.root","METPD17_RunD",step_size)

		#for testing
		#add met 2018 run C
		#reader.ProcessFileCNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/LLPSkims/condor_superclusters_defaultv9p2_MET_R18_AL1NpSC_v31_MET_AOD_Run2018C-15Feb2022_UL2018-v1.root","METPD18_RunC")
		#need to set all SC labels to 1 - pass as input arg
		#make_glgl_samples(reader_test)	
		#run for whole gluino + squark grids + ctau=30 gogoZ samples
	

	elif args.obj == "photon":
		reader_train.ProcessFileDNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_GJetsCR_EGamma_R18_InvMetPho30_NoSV_v31_EGamma_AOD_Run2018C.root","EGamma18_RunC",step_size=10000,maxnchunk=4000)
		#reader_train.ProcessFileDNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_diJetsCR_JetHT_R18_InvMET100_nolumimask_v31_JetHT_AOD_Run2018C-15Feb2022_UL2018-v1.root","JetHT18_RunC",step_size=10000,maxnchunk=1000)
		#reader_train.ProcessFileDNN("root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_diJetsCR_JetHT_R18_InvMET100_nolumimask_v31_JetHT_AOD_Run2018C-15Feb2022_UL2018-v1.root","JetHT18_RunC",step_size=10000,maxnchunk=1000)
	'''

	proc = ""
	if args.proc == "EGamma" and args.era == "C":
		if args.year == "2018":
			proc = "root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_GJetsCR_EGamma_R18_InvMetPho30_NoSV_v31_EGamma_AOD_Run2018C.root"
		else:
			print("Process and year not found")
			exit()
	if args.proc == "JetHT" and args.era == "C":
		if args.year == "2018":
			proc = "root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_diJetsCR_JetHT_R18_InvMET100_nolumimask_v31_JetHT_AOD_Run2018C-15Feb2022_UL2018-v1.root"
		else:
			print("Process and year not found")
			exit()
	
	if args.proc == "MET":
		print("these samples haven't been set yet....")
		exit()

	if args.test:
		reader = reader_test
	else:
		reader = reader_train

	if args.obj == "SC":
		reader.ProcessFileCNN(proc,step_size=10000,chunkrange = [args.chunkFirst, args.chunkLast])
	if args.obj == "photon"
		reader.ProcessFileDNN(proc,step_size=10000,chunkrange = [args.chunkFirst, args.chunkLast])



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--obj",help='objects to run over',choices=["photon","SC"],required=True)
	parser.add_argument("--objType",help='type of objects to run over',choices=["CMS","BHC","BHCPUCleaned"],default="CMS")
	parser.add_argument("--proc",help='process to use',choices=['EGamma','JetHT','GlGl','MET'],required=True)
	parser.add_argument("--era",required=True)
	parser.add_argument("--year",choices=['2017','2018'],required=True)
	parser.add_argument("--test",default=False,help="make test data",action='store_true')
	parser.add_argument("--chunkFirst",default=-1)
	parser.add_argument("--chunkLast",default=-1)
	args = parser.parse_args()
	main(args)
