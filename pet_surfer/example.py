#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Perform regional kinetic modeling (MRTM2) in an example subject using PetSurfer

Note:
    - Path to files and directory have to be modified according to data location
    - This code does not perform the segmentation of the reference region
"""

# Path to files and directory have to be modified according to data location

# Full path to BIDS PET data
base = 'test_subject'
base_dir = 'bids_directory'
pet_img = join(base_dir, base + '_pet.nii.gz')
json_file = join(base_dir, base + '_pet.json')

# Full path to directory containing FreeSurfer recon (SUBJECTS_DIR)
recon_dir = 'freesurfer_recons'  # i.e., full path to SUBJECTS_DIR
mid = 'test_subject'  # directory in SUBJECTS_DIR for the current subject

# Create output directory
out_dir = join('derivatives', 'pet_surfer', 'example_subject')
ps.assert_dir(out_dir)

#%% Create (weighted) average PET images

avg_img = join(out_dir, base + '_wavg.nii')
ps.create_weighted_average_pet(pet_img, json_file, avg_img, frames=range(9, 38)) # rames 10-38 with 0-based indexing

#%% Align PET & MR

reg = join(out_dir, base + '_to_anat.lta')
ref = join(recon_dir, mid, 'mri', 'norm.mgz')
aligned = join(out_dir, base + '_wavg_to_anat.nii')
ps.compute_align(avg_img, reg, ref, aligned=aligned)

# Visual QC for PET & MR coregistration
ps.visual_coreg_QC(ref, aligned, join(out_dir, 'QC_pet_anat_coreg.png'))

#%% Run gtmseg

ps.gtmseg(mid, recon_dir, options='--no-vermis --xcerseg')

# Visual QC for gtmseg
ps.visual_gtmseg_QC(
        join(recon_dir, mid),
        join(out_dir, 'QC_gtmseg.png')
    )

#%% Extract volume TACs

# Path to segmentation files - to be modified according to the actual location of the files
cereb_gm = join('derivatives', 'suit', 'cerebellar_gm.nii.gz'),  # reference region
gtmseg = join(recon_dir, mid, 'mri', 'gtmseg.mgz')  # gtmseg segmentation

# The labels_dct dictionary specifies the labels for extracting average TACs.
# The dictionary key is used as output file name. Additional information is 
# specified through the following fields:
#   file: path of the file containing the segmentation
#   ids: labels for which to extract the TACs. List containing int or lists of int.
#        Int will be considered as individual regions, and lists of int specify
#        regions formed by multiple labels. For example,
#        ids: [1, 2, [2, 3], [3, 4, 5]] will output the average TACs for labels
#        1, 2 and the combined region formed by the labels [2, 3] and [3, 4 ,5]. 
#   ext: extension of the output file. Depending on their usage, 
#        "nii.gz" or "dat" have to be used.

labels_dct = {
        'gtmseg-subcort': {
                'file': gtmseg,
                # Talamus, caudate, putamen, pallidum, hippocampus, amygdala
                'ids': [10, 11, 12, 13, 17, 18, 49, 50, 51, 52, 53, 54],
                'ext': 'nii.gz'
            },
        'cereb-gm': {
                'file': cereb_gm
                'ids': [1],
                'ext': 'dat'
            },
        'caudate-putamen': {
                'file':  gtmseg,
                'ids': [[11, 12, 50, 51]],  # indices in the inner list will be considered 
                'ext': 'nii.gz'
            }
    }

reg = join(out_dir, base + '_to_anat.lta')
ps.extract_vol_tacs(pet_img, gtmseg, reg, out_dir, labels_dct)

#%% Extract regional surface tacs from aparc

reg = join(out_dir, base + '_to_anat.lta')
ps.extract_surf_tacs(pet_img, mid, reg, recon_dir, out_dir)

#%% Compute MRTM model in high binding regions to obtain an estimate of k2'

# Create midframe timing file
mid_frames_dat = join(out_dir, 'midframes.sec.dat')
ps.create_mid_frame_dat(json_file, mid_frames_dat)

mrtm_hb_dir = join(out_dir, 'mrtm-hb')
hb_tac = join(out_dir, 'caudate-putamen.nii.gz')  # TAC of the high-binding region
ref = join(out_dir, 'cereb-gm.dat')  # TAC of the reference region
ps.mrtm(hb_tac, ref, mid_frames_dat, mrtm_hb_dir)

#%% Compute MRTM2 for target regions

mid_frames_dat = join(out_dir, 'midframes.sec.dat')  # timing file
mrtm_hb_dir = join(out_dir, 'mrtm-hb')  # for getting k2;
tac_types = ['annot_lh', 'annot_rh', 'gtmseg-subcort']  # parcellations
ref = join(out_dir, 'cereb-gm.dat')  # TAC of the reference region

for tac_type in tac_types:
    tacs = join(out_dir, tac_type + '.nii.gz')  # path to parcellation files
    mrtm2_dir = join(out_dir, 'mrtm2', tac_type)  # output directory
    ps.mrtm2(tacs, ref, mid_frames_dat, mrtm_hb_dir, mrtm2_dir)
