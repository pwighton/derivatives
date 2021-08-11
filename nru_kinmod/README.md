# Pharmacokinetic models
This directory contains the necessary MATLAB code to quantify the receptor availability using a variety of pharmacokinetic models. The models are largely implemented by Claus Svarer from the Neurobiology Research Unit, Copenhagen, Denmark.  

# Models and data
All the models are located in the `kinmod` directory, and test data can be found in the `test_data` directory. The test data includes the following:
1. *.tim file with the necessary timing information for all frames
2. *.sif file contains the timing information, including information about max true count information that can be used weight the data points, when applying the models.
3. KinMod.txt contains the time activity curves (TACs) for each given region (Bq/mL), including the timing information for each frame. The first column is the frame start time, the second column is the frame end time, the third column is the highbinding region (hb) used for reference tissue modeling (average between putamen and thalamus for DASB), the subsequent columns are the TACs for each given region, and the last column is the reference (ref) tissue TAC (cerebellum for DASB). 

# Running the models
The models can be run by entering e.g. `CreateMRTM2` in matlab. The user is then required to select the necessary files and parameters to estimate the model, and the model output is subsequently shown. 