import numpy as np
import argparse
from ConvertData import TTreeReader

def make_sms_samples_sc():
	#("root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-1500_mN2-500_mN1-100-ct0p1_superclusters_defaultv9p2.root","",step_size=10000,maxnchunk=-1,debug=False,labelas=1)
	path = "root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims_test/"
	glglz = ["SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-1500_mN2-500_mN1-100-ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-200-ct0p001_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-200-ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-200-ct0p3_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-350-ct0p001_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-350-ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1900_mN1-350-ct0p3_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-1950_mN1-1900-ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-200-ct0p001_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-200-ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-200-ct0p3_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-350-ct0p001_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-350-ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-2000_mN2-400_mN1-350-ct0p3_superclusters_defaultv9p2.root"]
	glglz = [path+i for i in glglz]

	glgl = ["SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-1500_mN2-500_mN1-100_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-250_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-500_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-1000_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-250_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-500_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1000_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1500_superclusters_defaultv9p2.root", "SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-250_superclusters_defaultv9p2.root", "SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-500_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1000_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1500_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1900_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-250_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-500_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-500_mN1-1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-500_mN1-250_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-GlGl_AODSIM_mGl-2500_mN2-1500_mN1-1000_superclusters_defaultv9p2.root"]
	glgl = [path+i for i in glgl]
	
	
	sqsq = ["SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1700_mN2-1500_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1700_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1850_mN2-1650_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-1850_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2000_mN2-1800_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2000_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2150_mN2-1950_mN1-100_ct0p1_superclusters_defaultv9p2.root","SMS_Sig_SVIPM100_v31_SMS-SqSq_AODSIM_mGl-2150_mN2-300_mN1-100_ct0p1_superclusters_defaultv9p2.root"]
	sqsq = [path+i for i in sqsq]
	return glgl, sqsq, glglz



def make_sms_samples_photons():
	path = "root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims_test/"
	glgl = ["SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-1500_mN2-500_mN1-100_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-1_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-250_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1000_mN1-500_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-1000_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-1_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-250_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1500_mN1-500_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1000_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1500_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-1_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-250_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1900_mN1-500_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1000_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1500_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1900_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-1_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-250_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-1950_mN1-500_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-500_mN1-1_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2000_mN2-500_mN1-250_photons_defaultv43_noIso.root","SMS_Sig_MET100_v31_SMS-GlGl_AODSIM_mGl-2500_mN2-1500_mN1-1000_photons_defaultv43_noIso.root"]
	glgl = [path+i for i in glgl]
	return glgl

def main(args):
	objType = "CMS"
	reader_train = TTreeReader(args.obj, objType)
	reader_test = TTreeReader(args.obj, objType,"test")
	step_size = 10000
	file = ""
	sample = ""
	if args.proc == "EGamma" and args.era == "C":
		if args.year == "2018":
			file = "root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_GJetsCR_EGamma_R18_InvMetPho30_NoSV_v31_EGamma_AOD_Run2018C.root"
			sample = "EGamma18_RunC"
		else:
			print("Process and year not found")
			exit()
	if args.proc == "JetHT" and args.era == "C":
		if args.year == "2018":
			file = "root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/condor_photons_defaultv4p3_noIso_diJetsCR_JetHT_R18_InvMET100_nolumimask_v31_JetHT_AOD_Run2018C-15Feb2022_UL2018-v1.root"
			sample = "JetHT18_RunC"
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

	labelas = -999
	if args.proc == "GlGl":
		labelas = 1
	if args.obj == "SC":
	#("root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-1500_mN2-500_mN1-100-ct0p1_superclusters_defaultv9p2.root","",step_size=10000,maxnchunk=-1,debug=False,labelas=1)
		debug = args.debug
		if args.proc == "GlGl":
			labelas = 1
		reader.ProcessFileCNN(file,sample,step_size=10000,chunkrange = [int(args.chunkFirst), int(args.chunkLast)],labelas=labelas)
	if args.obj == "photon":
	#("root://cmseos.fnal.gov//store/user/mlazarov/LLPMVA_TrainingSamples/LLPSkims/SMS_Sig_SVIPM100_v31_SMS-GlGlZ_AODSIM_mGl-1500_mN2-500_mN1-100-ct0p1_superclusters_defaultv9p2.root","",step_size=10000,maxnchunk=-1,debug=False,labelas=1)
		if args.proc == "GlGl":
			labelas = 4
			files = make_sms_samples_photons()
			sample = ""
			for file in files:
				reader.ProcessFileDNN(file,sample,step_size=10000,chunkrange = [int(args.chunkFirst), int(args.chunkLast)],labelas=labelas)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--obj",help='objects to run over',choices=["photon","SC"],required=True)
	parser.add_argument("--proc",help='process to use',choices=['EGamma','JetHT','GlGl','MET'],required=True)
	parser.add_argument("--era")
	parser.add_argument("--year",choices=['2017','2018'])
	parser.add_argument("--test",default=False,help="make test data",action='store_true')
	parser.add_argument("--chunkFirst",default=-1)
	parser.add_argument("--chunkLast",default=-1)
	args = parser.parse_args()
	main(args)
