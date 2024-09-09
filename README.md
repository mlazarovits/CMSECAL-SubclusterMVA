# CMSECAL-SubclusterMVA
Classifying subclusters of rechits from CMS ECAL in time and space

### Dependencies
- Install [Miniconda](https://docs.anaconda.com/miniconda/) to keep the python package installations separate from the system installation
- create a miniconda environment with the following packages installed
	- python (issues with latest version of TF and python - see Troubleshooting)
	- pandas
	- numpy
	- tensorflow
	- matplotlib
	- sklearn
	- shap
- build the conda environment with the associated packages by running
```
conda create -n [env_name] python pandas tensorflow
```
- activate the conda environment by running
```
conda activate [env_name]
```
- deactivate the conda environment with
```
conda deactivate
```
- if the `python` command is aliased via a login script, make sure to run scripts `python3`
- if the conda environment is already created, you can run `conda instal [pkg]` to install additional packages from within the virtual environment


### Data Parsing
- input: CSV file
`CSVReader` class in `ProcessData.py` is in charge of parsing CSV files. User needs to `CleanData()` and then extract the data with `GetData()`.

### Running neural network
From the input data, the user needs to pass the input data shape to `DeepNN` constructor. (Eventually will pass data when training network).
```
python3 runDNN.py
```

### Troubleshooting
Running in python3.12 with the latest version of tensorflow on MacOS Ventura 13.0 raised this error on the `model.fit()` call:
```
libc++abi: terminating with uncaught exception of type Xbyak::Error: x2APIC is not supported
```
This error indicates some issue with the tensorflow installation. Some cursory web-searching showed that this can happen with the versions of TF and python are not compatible. This can happen if the default version of python was used to create the environment and tensorflow was installed after environment creation, leading to incompatibilities. To ensure this doesn't happen, create the conda environment with all necessary packages,
```
conda create -n "env_name" python tensorflow pandas matplotlib scikit-learn
```
To be sure of the python version, you can create a conda environment with python3.11.X with the following command
```
conda create -n "env_name" python=3.11.X ipython tensorflow pandas matplotlib scikit-learn
```
solved this issue and the model was able to be fit without any errors raising. Here, `X` is a specific version number. 
