# MLJetReconstruction - using machine learning to reconstruct jets for CMS

The C++ data extraction code used here was based heavily on that foundv [here](https://github.com/cms-opendata-analyses/JetNtupleProducerTool).
For more information on the Implicit Quantile Networks see the folder IQN.

## Setting up
First, follow the instruction [here](http://opendata.cern.ch/docs/cms-guide-docker) to set up docker and install CMSSW. Make sure to get cmssw_10_6_8_patch1, not the version given in the tutorial.

Then enter the docker container and clone the repo and compile it. Make sure you are in the directory CMSSW_10_6_8_patch1/src before cloning. 
```
git clone https://github.com/alpha-davidson/IQNs-for-Jets.git
cd IQNS-for-Jets/JetAnalyzer
scram b                 # compiles the code
```

## Running the Mean Buildup Model

The first code that should be run is the dataset generation code. Do this by running
```
cmsRun python/ConfFile_cfg.py
```
This will result in the extraction of approximatley 3 million jets examples. Note that this can take on the order of 8 hours to run. After running, it will create the file mlData.txt. Transfer this to the python work area. To make sure you have all the required packages run
```
pip install numpy matplotlib tensorflow tables sklearn
```

After this you can start running the python programs. Start with
```
python cleanParticleData.py
```

After this, the the two networks can be trained. Do this using the code
```
python rawToGenQuanitle.py
python genToRecoQuantile.py
```
Note that both of these files have command line options which can be accessed with the --help flag. The defaults though are the values used currently.

Once these have been run, use
```
python runBatchPredictions.py
python basicDataExtraction.py
python plotBasicData.py
```
to generate the graphs for the rawToGen training direction.

Use
```
python runBatchPredictionsGenToReco.py
python basicDataExtractionGenToReco.py
python plotBasicDataGenToReco.py
```
to generate the graphs for the genToReco training direction.
