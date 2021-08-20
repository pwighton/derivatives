#! /usr/bin/env python

# -*- coding: utf-8 -*-
"""PetSurfer

Collection of functions and wrappers for performing the analysis of PET data
using FreeSurfer's PET pipeline (https://surfer.nmr.mgh.harvard.edu/fswiki/PetSurfer)
"""

import os
import nibabel as nib
import numpy as np
import json

from subprocess import Popen, PIPE
from os.path import join, isfile

#%% Utility functions


def assert_dir(dir_path):
    
    """
    Create directory, if it does not exist
    
    Arguments
    ---------
    dir_path: string
        path to directory to be create   
    """ 
    
    full_path = os.path.abspath(dir_path)
    if not os.path.isdir(full_path):
        print('Creating %s' % full_path)
        os.makedirs(full_path)


def run(cmd):
    
    """
    Excute a command
    
    Arguments
    ---------
    cmd: string
        command to be executed
    """ 
    
    print('\n' + cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    output, error = p.communicate()
    if output:
        print(output.decode('latin_1'))
    if error:
        print(error.decode('latin_1'))
    
    
#%% FreeSurfer wrapper
    

def mri_convert(fin, fout, options=''):
    
    """
    Wrapper for FreeSurfer's mri_convert
    
    Arguments
    ---------
    fin: string
        path to input file
    fout: string
        path to output file
    options:
        additional options (see mri_convert --help for details)
    """ 
    
    cmd = ' '.join(['mri_convert', fin, fout, options])
    run(cmd)


def mri_coreg(mov, ref, reg, options=''):
    
    """
    Wrapper for FreeSurfer's mri_coreg
    
    Arguments
    ---------
    mov: string
        path to moveable volume
    ref: string
        path to reference volume
    reg: string
        path to output registration file
    options:
        additional options (see mri_coreg --help for details)
    """ 
    
    cmd = ' '.join([
            'mri_coreg',
            '--mov', mov,
            '--ref', ref,
            '--reg', reg,
            options
        ])
    run(cmd)


def mri_vol2vol(mov, targ, fout, reg='--regheader', options=''):
        
    """
    Wrapper for FreeSurfer's mri_vol2vol
    
    Arguments
    ---------
    mov: string
        path to input volume
    targ: string
        path to target volume
    fout: string
        path to output volume
    reg: string
        path to registration file or --regheader (default)
    options:
        additional options (see mri_vol2vol --help for details)
    """ 
    
    if reg != '--regheader':
        reg = '--reg ' + reg
    
    cmd = ' '.join([
            'mri_vol2vol',
            '--mov', mov,
            '--targ', targ,
            '--o', fout,
            reg, options
        ])
    run(cmd)
    
    
def mri_vol2surf(mov, hemi, reg, fout,
                 projfrac=0.5,
                 fwhm=None,
                 options=''):
    
    """
    Wrapper for FreeSurfer's mri_vol2surf
    
    Arguments
    ---------
    mov: string
        path to input volume
    hemi: string
        hemisphere (lh or rh)
    reg: string
        path to registration file or --regheader (default)
    fout: string
        path to output volume
    projfrac: float
        projection fraction, default 0.5
    fwhm: None or int
        smoothing full width at half maximum in mm, default None (no smoothing)
    options:
        additional options (see mri_vol2surf --help for details)
    """ 
        
    cmd = ' '.join([
            'mri_vol2surf',
            '--mov', mov,
            '--hemi', hemi,
            '--reg', reg,
            '--o', fout,
            '--projfrac %f' % projfrac,
            options
        ])
    if fwhm is not None:
        cmd += ' '.join([cmd, '--surf-fwhm', '%i' % fwhm])
    run(cmd)


def mri_binarize(fin, fout, match, options):
    
    """
    Wrapper for FreeSurfer's mri_binarize
    
    Arguments
    ---------
    fin: string
        path to input volume
    fout: string
        path to output volume
    match: string
        list of labels to match, e.g., 2 21 32
    options:
        additional options (see mri_binarize --help for details)
    """     
    
    cmd = ' '.join([
            'mri_binarize',
            '--i', fin,
            '--o', fout,
            '--match', match,
            options
        ])
    run(cmd)
    
    
def mris_calc(f1, operation, fout, f2=''):
    
    """
    Wrapper for FreeSurfer's mri_calc
    
    Arguments
    ---------
    f1: string
        path to first input volume
    operation: string
        operation to perform (see mris_calc --help)
    fout: string
        path to output volume
    f2: string
        path to second input volume (optional)
    """    
    
    cmd = ' '.join(['mris_calc', '-o', fout, f1, operation, f2])
    run(cmd)
    

def gtmseg(subject, subjects_dir=None, options=''):
    
    """
    Wrapper for FreeSurfer's gtmseg
    
    Arguments
    ---------
    subject: string
        subject in SUBJECTS_DIR
    subjects_dir: string
        path to directory containing recons
    options:
        additional options (see mri_vol2surf --help for details)
    """    
    
    if subjects_dir is not None:
        os.environ['SUBJECTS_DIR'] = subjects_dir
        
    if not isfile(join(os.environ['SUBJECTS_DIR'], subject, 'mri', 'gtmseg.mgz')):
        cmd = ' '.join(['gtmseg', '--s', subject, options])
        # print(cmd)
        run(cmd)


#%% Specific processing

def get_timing(json_file):
    
    """
    Extract timing of dynamic PET data from JSON file 
    
    Arguments
    ---------
    json_file: string
        path to JSON file
    fout: string
        path to output average volume
    
    Return
    ---------
    frames_start: numpy array of float
        list of start time for each frame (sec) 
    frames_duration: numpy array of int
        list of frame duration (sec)
        
    """

    with open(json_file, 'r') as f:
        info = json.load(f)
    frames_start = np.array(info['FrameTimesStart'], dtype=float)
    frames_duration = np.array(info['FrameDuration'], dtype=float)
    return frames_start, frames_duration    

def create_weighted_average_pet(fin, json_file, fout, frames=None):
    
    """
    Create a time-weighted average of dynamic PET data using mid-frames
    
    Arguments
    ---------
    fin: string
        path to input dynamic PET volume
    fout: string
        path to output average volume
    frames: list of integers
        list of frames to be used for computing the average (indices are 0-based)
    """     
      
    if not isfile(fout):
        img = nib.load(fin)        
        data = img.get_fdata()

        frames_start, frames_duration = get_timing(json_file)
        
        # Check that the specified frame interval, if any, is valid
        if frames is None:
            frames = range(data.shape[-1])
        else:
            if frames[0] < 0:
                raise ValueError('The indice of of first frame needs to be equal or larger than 0')
            if frames[-1] >= data.shape[-1]:
                raise ValueError('The indice of of last frame needs to less than %i' % data.shape[-1])

        mid_frames = frames_start + frames_duration/2
        wavg = np.trapz(data[..., frames], dx=np.diff(mid_frames[frames]), axis=3)/np.sum(mid_frames)
        print('Saving average to ' + fout)
        nib.save(nib.Nifti1Image(wavg, img.affine), fout)
    else:
        print('File ' + fout + ' already exists. Skipping.')


def compute_align(mov, reg, ref, aligned=None):
    
    """
    Compute alignment between an average PET image and an anatomical image
    using mri_coreg
    
    Arguments
    ---------
    mov: string
        moveable volume to be aligned with anatomical image
    reg: string
        output registration file
    recon_dir:
        path to subject's FreeSurfer's recon directory
    aligned:
        path to average PET image aligned with anatomical image
    """      
        
    # Check if input image exists
    if not isfile(mov):
        raise ValueError('Mean image does not exist. ' + mov)

    # Check if reference image exists
    if not isfile(ref):
        raise ValueError('Reference image does not exist. ' + ref)
    
    # Run coregistration
    if not isfile(reg):
        mri_coreg(mov, ref, reg)

    # Align image to anat
    if aligned is not None:
        mri_vol2vol(mov, ref, aligned, reg=reg, options='--cubic')
        

def freeview_QC(cmd, png_concat, viewports=['sagittal', 'coronal', 'axial']):
    
    """
    Create images for quality control using freeview
    
    Arguments
    ---------
    cmd: string
        command to be executed by freeview
    png_concat: string
        path to file concatenating all views
    viewports: list of string
        views to be visualized (saggital, coronal, axial)
    """  
    
    # Handle strings
    if not isinstance(viewports, list):
        viewports = [viewports]
    
    pngs = []
    for viewport in viewports:
        png_out = join('/tmp', viewport + '.png')
        pngs += [png_out]
        cmd = ' '.join([cmd,
                '-viewport', viewport,
                '-ss ' + png_out,
                '-quit'
            ])
        run(cmd)
    
    # Concat images        
    cmd = ' '.join(['convert', ' '.join(pngs), '-append', png_concat])
    run(cmd)
    list(map(os.remove, pngs))  
    
def visual_coreg_QC(anat, pet, png):
    
    """
    Create images for quality control of the aligment between an average
    PET image and an anatomical image
    
    Arguments
    ---------
    anat: string
        path to anatomical image
    pet: string
        path to PET image
    png: string
        path to output image
    """  
    
    # Check if anat image exists
    if not isfile(anat):
        raise ValueError('Anatomical image does not exist. ' + anat)
        return

    # Check if reference image exists
    if not isfile(pet):
        raise ValueError('PET image does not exist. ' + pet)
            
    if not isfile(png):
        cmd = ' '.join([
                'freeview', anat,
                pet + ':colorscale=1000,4500:colormap=jet:opacity=0.3'
            ])
        freeview_QC(cmd, png)


def visual_gtmseg_QC(recon_dir, png):
    
    """
    Create images for quality control of the gtmseg segmentation
    
    Arguments
    ---------
    recon_dir: string
        path to the subject's recon directory
    png: string
        path to output image
    """  
    
    gtmseg = join(recon_dir, 'mri', 'gtmseg.mgz')
    
    if isfile(gtmseg) and not isfile(png):        
        anat = join(recon_dir, 'mri', 'norm.mgz')
        cmd = ' '.join([
                'freeview', anat,
                gtmseg + ':colormap=lut:opacity=0.3'
            ])
        freeview_QC(cmd, png)


def is_finished(tacs_dir, labels_dct):
    
    """
    Check wether all output TAC files have been created
    
    Arguments
    ---------
    tac_dir: string
        path to directory containing TAC files
    labels_dct: dictionary
        dictionary specifying TAC files to be created
    """ 
    
    keys = list(labels_dct.keys())
    ext = [labels_dct[key]['ext'] for key in keys]
    return np.all([isfile(join(tacs_dir, k + '.' + e))
                   for k, e in zip(keys, ext)])


def save_np_array_to_fs(X, fout):
    
    """
    Save surface data from numpy array to to Nifti
    
    Arguments
    ---------
    cmd: string
        command to be executed
    """  
    
    # Assert input shape
    if X.ndim > 2:
        raise ValueError('Input array has more than 2 dimensions')
    if X.ndim == 1:
        X = X.reshape([-1, 1])
    
    nib.save(nib.Nifti1Image(
                X.reshape([X.shape[0], 1, 1, X.shape[-1]]), np.eye(4)),
            fout)


def extract_vol_tacs(pet, targ, reg, out_dir, labels_dct):

    """
    Extract TACs from volume as specified in labels dictionary
    
    Arguments
    ---------
    pet: string
        path to input dynamic PET volume
    targ: string
        path to target volume in anatomical space (e.g., gtmseg.mgz)
    reg: string
        path to registration file (PET to anatomical space)
    out_dir: string
        path to output directory
    labels_dct: dictionary
        dictionary specifying TAC files to be created
    """     
    
    assert_dir(out_dir)

    # Assert tac files in gtmseg space
    tacs = join(out_dir, 'pet_to_gtmseg.nii.gz')
    if not is_finished(out_dir, labels_dct):
        mri_vol2vol(pet, targ, tacs, reg=reg, options='--nearest')

    print('Loading data...')
    data = nib.load(tacs).get_fdata()
                
    for key in labels_dct.keys():
        
        print('Processing ' + key)
        
        ext = labels_dct[key]['ext']        
        fout_data = join(out_dir, key + '.' + ext)
        fout_dct = join(out_dir, key + '.npy')
        if not isfile(fout_data) or not isfile(fout_dct):

            labels = nib.load(labels_dct[key]['file']).get_fdata()
            ids = labels_dct[key]['ids']
            
            # Extract mean tacs
            mean_tacs = []
            for label in ids:
                mask = np.isin(labels, label)
                mean_tacs += [np.mean(data[mask, :], axis=0)]
            mean_tacs = np.vstack(mean_tacs)
                
            
            print(fout_data)
            if ext == 'nii.gz':
                save_np_array_to_fs(mean_tacs, fout_data)
            elif ext == 'dat':
                np.savetxt(fout_data, mean_tacs.T, fmt='%0.8f')
            else:
                raise ValueError('Invalid extension ' + ext)
            print(fout_dct)
            np.save(fout_dct, labels_dct[key])
            
    # Clean up, save space
    if isfile(tacs):
        os.remove(tacs)


def extract_surf_tacs(pet, mid, reg, recon_dir, out_dir):

    """
    Extract TACs from surface as specified FreeSurfer parcellations
    
    Arguments
    ---------
    pet: string
        path to input dynamic PET volume
    mid: string
        subject name in the recon directory (i.e., SUBJECTS_DIR)
    reg: string
        path to registration file (PET to anatomical space)
    recon_dir: string
        path to directory containing recons, will be assigned to SUBJECTS_DIR
    out_dir: string
        path to output directory
    """     
        

    assert_dir(out_dir)

    for hemi in ['lh', 'rh']:

        fout = join(out_dir, 'annot_' + hemi)
        if not isfile(fout + '.nii.gz') or not isfile(fout + '.npy'):
            
            tacs = join(out_dir, 'tacs_' + hemi + '.nii.gz')
            if not isfile(tacs):
                os.environ['SUBJECTS_DIR'] = recon_dir
                mri_vol2surf(
                        pet, hemi, reg, tacs,
                        options='--srcsubject ' + mid + ' --cortex'
                    )

            data = nib.load(tacs).get_fdata()
            annot = join(recon_dir, mid, 'label', hemi + '.aparc.annot')
            labels, ctab, names = zip(nib.freesurfer.read_annot(annot))
            ids = np.unique(labels)
            ids = ids[ids != -1]
            
            # Extract mean tacs
            mean_tacs = []
            for label in ids:
                mask = np.isin(labels, label).reshape(-1)
                mean_tacs += [np.mean(data[mask, ...], axis=0).reshape(-1)]
            mean_tacs = np.vstack(mean_tacs)
        
            print(fout + '.nii.gz')        
            save_np_array_to_fs(mean_tacs, fout + '.nii.gz')
            labels_dct = {'file': annot, 'ids': ids}
            print(fout + '.npy')
            np.save(fout + '.npy', labels_dct)   

            # Clean up, save space
            if isfile(tacs):
                os.remove(tacs)


def create_mid_frame_dat(json_file, fout):

    """
    Extract timing of mid frames of dynamic PET data and save for usage by
    mri_glmfit
    
    Arguments
    ---------
    json_file: string
        path to BIDS json PET file
    fout: string
        path to output file
    """  

    if not isfile(fout):
        frames_start, frames_duration = get_timing(json_file)
        mid_frames = frames_start + frames_duration/2    
        np.savetxt(fout, mid_frames, fmt='%0.1f')
    else:
        print('Midframe timing file already exists. Skipping.')


def mrtm(tacs, ref, timing, out_dir):
    
    """
    Perform MRTM modeling using FreeSurfer's mri_glmfit
    
    Arguments
    ---------
    tacs: string
        path to input PET TACs file
    ref: string
        path to reference TAC file
    timing: string
        path to timing file
    out_dir: string
        path to output directory
    """  
 
    cmd = ' '.join([
            'mri_glmfit', 
            '--y', tacs,
            '--mrtm1', ref, timing,
            '--o', out_dir,
            '--no-est-fwhm --nii.gz --yhat-save'
        ])
    run(cmd)
    
    
def mrtm2(tacs, ref, timing, mrtm_dir, out_dir):
    
    """
    Perform MRTM2 modeling using FreeSurfer's mri_glmfit
    
    Arguments
    ---------
    pet: string
        path to input PET TACs file
    ref: string
        path to reference TAC file
    timing: string
        path to timing file
    out_dir: string
        path to output directory
    """  
    
    assert_dir(out_dir)
    
    # Load MRTM parameters
    k2p = '%0.15f' % np.loadtxt(join(mrtm_dir, 'k2prime.dat'))
                   
    cmd = ' '.join([
            'mri_glmfit',
            '--y', tacs,
            '--mrtm2', ref, timing, k2p,
            '--o', out_dir,
            '--no-est-fwhm --nii.gz --yhat-save'
        ])
    run(cmd)