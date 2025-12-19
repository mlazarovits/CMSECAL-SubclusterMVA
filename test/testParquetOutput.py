import dask.dataframe as dd

parquet_path = "parquet_output/CMS_photons_test/SMS_GlGl/"
file = parquet_path+"chunk_00005_sample_SMS_GlGl_mGl_2000_mN2_1950_mN1_1500_type_CMS_Photons.parquet"
ddf = dd.read_parquet(file)
df = ddf.compute()
labelcol = 'Photon_trueLabel_CMS'
print("labels",df[labelcol].unique())
