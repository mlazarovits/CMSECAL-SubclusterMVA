from ProcessData import DataCleaner

#testing remote parquet file on LPC
#root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/test_chunk_00000_sample_METPD18_RunB_type_CMS.parquet
#parquet_path = "root://cmseos.fnal.gov//store/user/malazaro/LLPMVA_TrainingSamples/"
parquet_path = "testparquet/"
cleaner = DataCleaner(parquet_path, "SC", "CMS", printStats = True)
dask_df = cleaner.GetDaskData(debug=False)
cleaner.CleanAndConvert(dask_df)
#cleaner.SelectClass(1,"EGamma")
cleaner.BarrelOnly("SC_EtaCenter")
