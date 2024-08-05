# CMSECAL-SubclusterMVA
Classifying subclusters of rechits from CMS ECAL in time and space

### Dependencies
- Install [Miniconda](https://docs.anaconda.com/miniconda/) to keep the python package installations separate from the system installation
- create a miniconda environment with the following packages installed
	- python
	- pandas
	- numpy
	- tensorflow
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
python runDNN.py
```
