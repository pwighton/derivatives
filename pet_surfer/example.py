#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Perform regional kinetic modeling (MRTM2) in an example subject using PetSurfer
"""

import os
import re
import importlib
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
from os.path import join, isdir, isfile
from multiprocessing import Pool
from itertools import product

os.getcwd()
os.chdir('/data1/vbeliveau/PetSurfer')

import petsurfer
importlib.reload(petsurfer)

ps_dir = join(os.getcwd(), 'derivatives', 'pet_surfer')
petsurfer.assert_dir(ps_dir)

# base = 'sub-sub-11155_ses-baseline'
# base_dir = 'sub-11155/ses-baseline/pet/'
# pet_img = 'sub-11155/ses-baseline/pet/sub-11155_ses-baseline_pet.nii'
# json_file = 'sub-11155/ses-baseline/pet/sub-11155_ses-baseline_pet.json'

base = 'sb123'
base_dir = 'sb123'
pet_img = join(base, base + '_pet.nii.gz')
json_file = join(base_dir, base + '_pet.json')
mid = 'f6885_GD'

out_dir = join(ps_dir, base_dir)
petsurfer.assert_dir(out_dir)


#%% Create (weighted) average PET images

avg_img = join(out_dir, base + '_wavg.nii')
petsurfer.create_weighted_average_pet(pet_img, json_file, avg_img, frames=range(0, 35))

#%% Align PET & MR

reg = join(out_dir, base + '_to_anat.lta')
ref = join('recons', 'f6885_GD', 'mri', 'norm.mgz')
aligned = join(out_dir, base + '_wavg_to_anat.nii')
petsurfer.compute_align(avg_img, reg, ref, aligned=aligned)

# Visual QC for PET & MR coregistration
petsurfer.visual_coreg_QC(ref, aligned, join(out_dir, 'QC_pet_anat_coreg.png'))

#%% Run gtmseg

recon_dir = join(os.getcwd(), 'recons')
petsurfer.gtmseg(mid, recon_dir, options='--no-vermis --xcerseg')

# Visual QC for gtmseg
petsurfer.visual_gtmseg_QC(
        join(recon_dir, mid),
        join(out_dir, 'QC_gtmseg.png')
    )

#%% Run SUIT

anat = join('recons', mid, 'mri', 'norm.mgz')
aparc_aseg = join('recons', mid, 'mri', 'aparc+aseg.mgz')
suit_dir = join(out_dir, 'suit')
petsurfer.run_suit(anat, aparc_aseg, suit_dir)

# Visual QC for SUIT
petsurfer.visual_suit_QC(suit_dir, join(out_dir, 'QC_suit.png'))

#%% Label cerebellar GM as reference region, excluding vermis

gtmseg = join('recons', mid, 'mri', 'gtmseg.mgz')
suit_dir = join(out_dir, 'suit')
petsurfer.label_cerebellar_gm(gtmseg, suit_dir)

#%% Extract volume TACs

gtmseg = join('recons', mid, 'mri', 'gtmseg.mgz')

labels_dct = {
        'gtmseg-subcort': {
                'file': gtmseg,
                # Talamus, caudate, putamen, pallidum, hippocampus, amygdala
                'ids': [10, 11, 12, 13, 17, 18, 49, 50, 51, 52, 53, 54],
                'ext': 'nii.gz'
            },
        'cereb-gm': {
                'file': join(out_dir, 'suit', 'cerebellar_gm.nii.gz'),
                'ids': [1],
                'ext': 'dat'
            },
        'caudate-putamen': {
                'file':  gtmseg,
                'ids': [[11, 12, 50, 51]],
                'ext': 'nii.gz'
            }
    }

reg = join(out_dir, base + '_to_anat.lta')
petsurfer.extract_vol_tacs(pet_img, gtmseg, reg, out_dir, labels_dct)

#%% Extract regional surface tacs from aparc

reg = join(out_dir, base + '_to_anat.lta')
petsurfer.extract_surf_tacs(pet_img, mid, reg, 'recons', out_dir)

#%% Compute MRTM in high binding regions

# Create midframe timing file
mid_frames_dat = join(out_dir, 'midframes.sec.dat')
petsurfer.create_mid_frame_dat(json_file, mid_frames_dat)

mrtm_hb_dir = join(out_dir, 'mrtm-hb')
hb_tac = join(out_dir, 'caudate-putamen.nii.gz')
ref = join(out_dir, 'cereb-gm.dat')
petsurfer.mrtm(hb_tac, ref, mid_frames_dat, mrtm_hb_dir)

#%% Compute MRTM2 for target regions

mid_frames_dat = join(out_dir, 'midframes.sec.dat')
mrtm_hb_dir = join(out_dir, 'mrtm-hb')
tac_types = ['annot_lh', 'annot_rh', 'gtmseg-subcort']
ref = join(out_dir, 'cereb-gm.dat')

for tac_type in tac_types:
    tacs = join(out_dir, tac_type + '.nii.gz')
    mrtm2_dir = join(out_dir, 'mrtm2', tac_type)
    petsurfer.mrtm2(tacs, ref, mid_frames_dat, mrtm_hb_dir, mrtm2_dir)
