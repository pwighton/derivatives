# Processing of PET data with FreeSurfer (PET Surfer)

This repository contains python code for the processing of PET data using the PET Surfer pipeline (https://surfer.nmr.mgh.harvard.edu/fswiki/PetSurfer). Currently, only regional analyis with MRTM and MRTM2 models are implemented.

An example is included in **example.py** and decribes how to perform the regional analysis of a [11C]DASB dataset. A example dataset can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds001421) (), but the corresponding FreeSurfer reconstruction and label for reference region currently have to be provided by the user.  

## References

PET Surfer:
Greve DN, Svarer C, Fisher PM, et al. Cortical surface-based analysis reduces bias and variance in kinetic modeling of brain PET data. Neuroimage. 2014;92C:225-236. doi:10.1016/j.neuroimage.2013.12.021

MRTM:
Ichise M, Liow J-S, Lu J-Q, et al. Linearized Reference Tissue Parametric Imaging Methods: Application to [11C]DASB Positron Emission Tomography Studies of the Serotonin Transporter in Human Brain. J Cereb Blood Flow Metab. 2003;23(9):1096-1112. doi:10.1097/01.WCB.0000085441.37552.CA

MRTM2:
Ichise M, Liow J-S, Lu J-Q, et al. Linearized Reference Tissue Parametric Imaging Methods: Application to [11C]DASB Positron Emission Tomography Studies of the Serotonin Transporter in Human Brain. J Cereb Blood Flow Metab. 2003;23(9):1096-1112. doi:10.1097/01.WCB.0000085441.37552.CA