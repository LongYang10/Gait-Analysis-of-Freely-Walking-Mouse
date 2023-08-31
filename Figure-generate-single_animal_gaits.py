# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:31:16 2023

@author: lonya
"""
#%% preset
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import messagebox
import pickle
import os
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import numpy as np
from scipy.stats import variation 
#%% Select animals data
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Choose: tagging eMouse.mat files',\
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
filez = fd.askopenfilenames(parent=root, title='Choose: all_mice_datapath.pickle')
root.destroy()
print(filez)
with open(filez[0], 'rb') as f:
    all_files = pickle.load(f)
#%% processing data
mouse_id = []
group_id = []
strain = []
stage = []
mouse_gait = pd.DataFrame()
mouse_mean_d = pd.DataFrame()
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
    
    # get kinematics
    params['start_stop_minimum']=0.3
    kinematics = mf.mouse_kinematics(mouse2D, params)
    
    # calculate head angle
    nose_tail_v = mouse2D['nose']-mouse2D['tail']
    params['minimum_turn'] = 20
    head_angles, turn_labels = mf.walk_angle_change(nose_tail_v, \
                                                    kinematics['bd_start_stop'],\
                                                    params)
    
    # calculate self-view mouse, gait cycle
    svm = mf.createSVM(mouse2D)
    svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')
    
    # get gaits & strides
    gait, stride = mf.calculate_gait(svm_bf, mouse2D) #1/11/2023
    gait_trimed, stride_trimed = mf.trim_gait(gait, stride)
    
    # calculate the immobile time
    camTimes = params['camTimes']
    bd_speed = kinematics['bd_speed'][int(np.round(camTimes[0]*params['fr'])):\
                                      int(np.round(camTimes[1]*params['fr']))]
    immobile_th = 2 # speed<2mm/s
    immobile_n = len(np.where(bd_speed<immobile_th)[0])
    immobile_t = immobile_n/params['fr']
    immobile_pct = immobile_t/(camTimes[1]-camTimes[0])*100
    
    # calculate the walking rate: number of walking bouts per minute
    walk_rate = kinematics['bd_start_stop'].shape[0]/(camTimes[1]-camTimes[0])*60 #second to min
    
    # calculate the walking distance
    walk_distance = mf.walk_distance(mouse2D['body'], kinematics['bd_start_stop'], params)
    
    # calculate mean walk speed
    walk_speed = mf.walk_speed(kinematics['bd_speed'], kinematics['bd_start_stop'], params)
    
    # calculate limb phase angle
    svm_angle = mf.get_mouse_angle_pct(svm_bf, stride_trimed)
    
    #calculate limb coordination
    lf_lr, lf_rf, lf_rr, rf_rr, rf_lr, lr_rr = \
        mf.limb_coord(svm, walkTimes, params)
    # get full stride
    full_stride, _ = mf.get_full_stride(stride, walkTimes)
    temp_stride = np.asarray(full_stride)
    temp_stride = np.reshape(temp_stride, (int(len(full_stride)/12),12))

    lr_stance = temp_stride[:,0]
    lf_stance = temp_stride[:,3]
    rf_stance = temp_stride[:,6]
    rr_stance = temp_stride[:,9]

    # find the corresponding phase angle at each stance time, reference to lr
    lr_index = np.round(lr_stance*params['fr']).astype(int)
    lr_stance_p = svm_angle['lr'][lr_index]
    lf_index = np.round(lf_stance*params['fr']).astype(int)
    lf_stance_p = svm_angle['lr'][lf_index]
    rf_index = np.round(rf_stance*params['fr']).astype(int)
    rf_stance_p = svm_angle['lr'][rf_index]
    rr_index = np.round(rr_stance*params['fr']).astype(int)
    rr_stance_p = svm_angle['lr'][rr_index]

    lr_lf_m, _,_ = mf.circ_m(lf_stance_p)
    lr_lr_m, _,_ = mf.circ_m(lr_stance_p)
    lr_rf_m, _,_ = mf.circ_m(rf_stance_p)
    lr_rr_m, _,_ = mf.circ_m(rr_stance_p)
    
    mouse_mean = {
        'mouse_id': mouse_id[-1],
        'strain': strain[-1],
        'group_id': group_id[-1],
        'head_angle': np.nanmean(head_angles),
        'immobile_pct': immobile_pct,
        'walk_rate': walk_rate,
        'walk_distance': np.sum(walk_distance),
        'walk_speed': np.mean(walk_speed),
        'stride_speed_lf': np.mean(gait_trimed['lf']['stride_velocity']),
        'stride_speed_lr': np.mean(gait_trimed['lr']['stride_velocity']),
        'stride_speed_rf': np.mean(gait_trimed['rf']['stride_velocity']),
        'stride_speed_rr': np.mean(gait_trimed['rr']['stride_velocity']),
        'stride_length_lf': np.mean(gait_trimed['lf']['stride_length']),
        'stride_length_lr': np.mean(gait_trimed['lr']['stride_length']),
        'stride_length_rf': np.mean(gait_trimed['rf']['stride_length']),
        'stride_length_rr': np.mean(gait_trimed['rr']['stride_length']),
        'stride_dur_lf': np.mean(gait_trimed['lf']['swing_duration']+\
                                 gait_trimed['lf']['stance_duration']),
        'stride_dur_lr': np.mean(gait_trimed['lr']['swing_duration']+\
                                 gait_trimed['lr']['stance_duration']),
        'stride_dur_rf': np.mean(gait_trimed['rf']['swing_duration']+\
                                 gait_trimed['rf']['stance_duration']),
        'stride_dur_rr': np.mean(gait_trimed['rr']['swing_duration']+\
                                 gait_trimed['rr']['stance_duration']),
        'sw_st_ratio_lf': np.mean(gait_trimed['lf']['swing_stance_ratio']),
        'sw_st_ratio_lr': np.mean(gait_trimed['lr']['swing_stance_ratio']),
        'sw_st_ratio_rf': np.mean(gait_trimed['rf']['swing_stance_ratio']),
        'sw_st_ratio_rr': np.mean(gait_trimed['rr']['swing_stance_ratio']),
        'stride_speed_cv_lf': variation(gait_trimed['lf']['stride_velocity']),
        'stride_speed_cv_lr': variation(gait_trimed['lr']['stride_velocity']),
        'stride_speed_cv_rf': variation(gait_trimed['rf']['stride_velocity']),
        'stride_speed_cv_rr': variation(gait_trimed['rr']['stride_velocity']),
        'stride_length_cv_lf': variation(gait_trimed['lf']['stride_length']),
        'stride_length_cv_lr': variation(gait_trimed['lr']['stride_length']),
        'stride_length_cv_rf': variation(gait_trimed['rf']['stride_length']),
        'stride_length_cv_rr': variation(gait_trimed['rr']['stride_length']),
        'stride_dur_cv_lf': variation(gait_trimed['lf']['swing_duration']+\
                                 gait_trimed['lf']['stance_duration']),
        'stride_dur_cv_lr': variation(gait_trimed['lr']['swing_duration']+\
                                 gait_trimed['lr']['stance_duration']),
        'stride_dur_cv_rf': variation(gait_trimed['rf']['swing_duration']+\
                                 gait_trimed['rf']['stance_duration']),
        'stride_dur_cv_rr': variation(gait_trimed['rr']['swing_duration']+\
                                 gait_trimed['rr']['stance_duration']),
        'sw_st_ratio_cv_lf': variation(gait_trimed['lf']['swing_stance_ratio']),
        'sw_st_ratio_cv_lr': variation(gait_trimed['lr']['swing_stance_ratio']),
        'sw_st_ratio_cv_rf': variation(gait_trimed['rf']['swing_stance_ratio']),
        'sw_st_ratio_cv_rr': variation(gait_trimed['rr']['swing_stance_ratio']),
        'lf_lr_r':np.mean(lf_lr),
        'lf_rf_r':np.mean(lf_rf),
        'lf_rr_r':np.mean(lf_rr),
        'rf_rr_r':np.mean(rf_rr),
        'rf_lr_r':np.mean(rf_lr),
        'lr_rr_r':np.mean(lr_rr),
        'lr_lf_dph': lr_lf_m,
        'lr_lr_dph': lr_lr_m,
        'lr_rf_dph': lr_rf_m,
        'lr_rr_dph': lr_rr_m
        }
    
    mouse_mean_df = pd.DataFrame(data=mouse_mean,index=[1])
    mouse_mean_d = pd.concat([mouse_mean_d,mouse_mean_df], \
                                 ignore_index=True)
    
#%% create path for figures
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']
output_path = all_files[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% save data
print('data saved to: ' + output_path)

with open(output_path + 'all-single-mice-gaits.pickle', 'wb') as f:
    pickle.dump(mouse_mean_d, f)
#Save gaits data into excel
mouse_mean_d.to_csv(output_path + 'all-single-mice-gaits.csv')
