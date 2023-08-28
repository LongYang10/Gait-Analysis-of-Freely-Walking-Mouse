# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:11:06 2023

@author: lonya
"""
#%% Load Data
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import messagebox
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import pickle
import os
#%% Select animals data
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Choose tagging eMouse.mat files',\
                                filetypes = (("matlab files","*.mat"),\
                                             ("all files","*.*")))
    msgbox = messagebox.askquestion ('Add files','add extra files',\
                                     icon = 'warning')
    return list(files), msgbox
files, msgbox = main()
all_files = files
while msgbox =='yes':
    files_2, msgbox = main()
    for i in files_2:
        files.append(i)
root.destroy()
#%% Load previous saved data path
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, title='Select all_mice_datapath.pickle')
root.destroy()
print(filez)

with open(filez[0], 'rb') as f:
    all_files = pickle.load(f)
    
#%% processing data
mouse_id = []
group_id = []
strain = []
stage = []
all_HL = pd.DataFrame()
all_binning_fr = np.empty((31,1))
all_psth_start = np.empty((1,999))
all_psth_stop = np.empty((1,999))
all_start_p = np.empty((1,))
all_stop_p = np.empty((1,))
all_rval = np.empty((1,))
all_pval = np.empty((1,))
all_lf = np.empty((1,24))
all_lr = np.empty((1,24))
all_rf = np.empty((1,24))
all_rr = np.empty((1,24))
all_lf_va = np.empty((1,)) #save mean phase angle of each neuron
all_lr_va = np.empty((1,))
all_rf_va = np.empty((1,))
all_rr_va = np.empty((1,))
all_start_speed = np.empty((1,161))
all_stop_speed = np.empty((1,161))
all_group = np.empty((1,))
all_d1_d2 = np.empty((1,))

for file in all_files:
    print('load file: ' + file)
    backslash_id = [m for m,x in enumerate(file) if x=='/']
    mouse_id.append(file[backslash_id[-3]+1:backslash_id[-2]])
    strain.append(file[backslash_id[-4]+1:backslash_id[-3]])
    group_id.append(file[backslash_id[-5]+1:backslash_id[-4]])
    stage.append(file[backslash_id[-2]+1:backslash_id[-1]])
    
    matfile = sio.loadmat(file, squeeze_me = True)
    eMouse = matfile['eMouse']
    # get locomotion trajectory
    mouse2D = mf.mouse2D_format(eMouse)
    walkTimes = mouse2D['walkTimes']
    params = mouse2D['params']
    
    # calculate self-view mouse, gait cycle
    svm = mf.createSVM(mouse2D)
    
    # get filtered self-view mouse
    svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')
    #svm_angle = mf.get_mouse_angle(svm_bf)
    
    # get gaits & strides
    gait, stride = mf.calculate_gait(svm_bf, mouse2D) #1/11/2023
    gait_trimed, stride_trimed = mf.trim_gait(gait, stride)
    
    # get kinematics
    kinematics = mf.mouse_kinematics(mouse2D, params)
    
    # get ephys data
    st_mtx, camTimes, tagging, cln, clwf, fs, xcoords, ycoords, unitBestCh = \
        mf.ephys_format(eMouse)
    
    # calculate binning speed and firing rate
    params['speed_space'] = 10    # default phase bin size 15 degree
    params['speed_bin_valid_th'] = 0.005 # default at least 0.5% data included
    params['maximum_speed'] = 310 # default 300mm/s
    params['minimum_speed'] = 0 #
    params['window'] = [0, params['rec_duration']]
    params['remove_bad_bins'] = False
    binning_speed, binning_fr, binning_se = \
        mf.calculate_speed_score(kinematics['bd_speed'],st_mtx, params)
    rval, pval = np.zeros(binning_fr.shape[1]), np.zeros(binning_fr.shape[1])
    for i in range(binning_fr.shape[1]):
        rval[i],pval[i] = pearsonr(binning_speed, binning_fr[:,i])

    #calculate the firing rate around start/stop of all neurons
    params['start_stop_win'] = [-10,10]
    params['sig_win'] = [-0.5,0.5]
    params['binsize'] = 0.02
    t, psth_start, start_se, start_p,_  = \
        mf.psth_norm(kinematics['bd_start_stop'][:,0], \
                     st_mtx, np.ones(len(st_mtx)), params) #start times
    _, psth_stop, stop_se, stop_p,_ = \
        mf.psth_norm(kinematics['bd_start_stop'][:,1], \
                     st_mtx, np.ones(len(st_mtx)), params) #stop times
    #calculate the body speed around start/stop
    bd = kinematics['bd_speed']
    upt = kinematics['bd_start_stop'][:,0]
    win = [-1, 1.0125]
    fr = params['fr']
    bins = np.arange(win[0], win[1], 1/fr)
    start_speed = np.zeros((upt.shape[0], len(bins)))
    for i in range(upt.shape[0]):
        e = upt[i]
        index = ((bins+e)*fr).astype(int)
        if max(index)<len(bd):
            start_speed[i,:] = bd[index]
    downt = kinematics['bd_start_stop'][:,1]
    stop_speed = np.zeros((downt.shape[0], len(bins)))
    for i in range(downt.shape[0]):
        e = downt[i]
        index = ((bins+e)*fr).astype(int)
        if max(index)<len(bd):
         stop_speed[i,:] = bd[index]
        
    # calculate phase locking
    params['n_circ_bins'] = 24
    params['density'] = False
    params['norm']=True
    
    # calculate limb phase angle
    svm_angle = mf.get_mouse_angle_pct(svm_bf, stride_trimed)
    
    #calculate phase distribution during stride
    svm_stride_angle = mf.get_stride_angle(svm_angle, stride_trimed)
    svm_stride_h, _= mf.get_stride_hist(svm_stride_angle, params)
    # pick out the phase angle when neuron firing during stride
    svm_sp_mtx_str = mf.get4limb_spa_stride(svm_angle, st_mtx, stride_trimed)
    # calculate the distribution of spike phase
    svm_sp_hmtx_str = mf.get4limb_spah_stride(svm_sp_mtx_str, svm_stride_h, \
                                              params)
    svm_spike_r_str = mf.get4limb_spa_circ_stride(svm_sp_mtx_str, \
                                                  svm_stride_h, params)
        
    # save labels
    if group_id[-1] == 'HL':
        group_label = np.zeros(len(st_mtx))
    elif group_id[-1] == 'CT':
        group_label = np.ones(len(st_mtx))
    elif group_id[-1] == 'PD':
        group_label = 2*np.ones(len(st_mtx))
    else:
        group_label = 3*np.ones(len(st_mtx))
        
    strain_label = np.zeros(len(st_mtx))
    if strain[-1] == 'D1':
        strain_label[np.where(tagging == 1)[0]] = 1
    elif strain[-1] == 'D2':
        strain_label[np.where(tagging == 1)[0]] = 2
    else:
        print('no tagging data')
        
    # for multiple mice
    all_binning_fr = np.append(all_binning_fr, binning_fr, axis=1)
    all_psth_start = np.append(all_psth_start, psth_start, axis=0)
    all_psth_stop = np.append(all_psth_stop, psth_stop, axis=0)
    all_start_p = np.append(all_start_p, start_p)
    all_stop_p = np.append(all_stop_p, stop_p)
    all_rval = np.append(all_rval, rval)
    all_pval = np.append(all_pval, pval)
    all_lf = np.append(all_lf, svm_sp_hmtx_str['lf'], axis=0)
    all_lr = np.append(all_lr, svm_sp_hmtx_str['lr'], axis=0)
    all_rf = np.append(all_rf, svm_sp_hmtx_str['rf'], axis=0)
    all_rr = np.append(all_rr, svm_sp_hmtx_str['rr'], axis=0)
    all_lf_va = np.append(all_lf_va, svm_spike_r_str['lf'][:,1])
    all_lr_va = np.append(all_lr_va, svm_spike_r_str['lr'][:,1])
    all_rf_va = np.append(all_rf_va, svm_spike_r_str['rf'][:,1])
    all_rr_va = np.append(all_rr_va, svm_spike_r_str['rr'][:,1])
    all_start_speed = np.append(all_start_speed, start_speed, axis=0)
    all_stop_speed = np.append(all_stop_speed, stop_speed, axis=0)
    all_group = np.append(all_group, group_label)
    all_d1_d2 = np.append(all_d1_d2, strain_label)
    
    healthy_df = mf.get_healthy_df(rval, pval, start_p, stop_p, \
                                   mouse_id[-1], strain[-1])
    all_HL = pd.concat([all_HL, healthy_df], ignore_index = True)
    
all_binning_fr = np.delete(all_binning_fr, 0, 1)
all_psth_start = np.delete(all_psth_start, 0, 0)
all_psth_stop = np.delete(all_psth_stop, 0, 0)
all_start_p = np.delete(all_start_p,0,None)
all_stop_p = np.delete(all_stop_p,0,None)
all_rval = np.delete(all_rval,0,None)
all_pval = np.delete(all_pval,0,None)
all_lf = np.delete(all_lf, 0, 0)
all_lr = np.delete(all_lr, 0, 0)
all_rf = np.delete(all_rf, 0, 0)
all_rr = np.delete(all_rr, 0, 0)
all_lf_va = np.delete(all_lf_va,0,None)
all_lr_va = np.delete(all_lr_va,0,None)
all_rf_va = np.delete(all_rf_va,0,None)
all_rr_va = np.delete(all_rr_va,0,None)
all_start_speed = np.delete(all_start_speed, 0, 0)
all_stop_speed = np.delete(all_stop_speed, 0, 0)
all_group = np.delete(all_group, 0, None)
all_d1_d2 = np.delete(all_d1_d2,0, None)

#%% create path for figures
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']
output_path = all_files[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% Save data analysis results
print('data saved to: ' + output_path)
save_varaibles = (all_binning_fr,\
                  all_psth_start,\
                  all_psth_stop,\
                  all_start_p, all_stop_p,\
                  all_rval, all_pval, \
                  all_lf, all_lr, all_rf, all_rr,\
                  all_lf_va, all_lr_va, all_rf_va, all_rr_va,\
                  all_start_speed, all_stop_speed,\
                binning_speed, all_group, all_d1_d2)
with open(output_path + 'all-single-neuron-mtx.pickle', 'wb') as f:
    pickle.dump(save_varaibles, f)