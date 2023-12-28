# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:31:16 2023

@author: lonya
"""
'''
######################################## 
preset
########################################
'''
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import messagebox
import pickle
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import numpy as np
from scipy.stats import variation 
import os

'''
#########################################
Load Data (all laser manipulation animal)
#########################################
''' 
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                        title='Choose eMouse.mat files in laser stim group',\
                                        filetypes = (("matlab files","*.mat"),("all files","*.*")))
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

'''
######################################### 
processing data
save all gait parameters into 3 catagories: pre-laser, laser, post-laser
#########################################
'''
mouse_id = []
group_id = []
strain = []
stage = []
mouse_gait = pd.DataFrame()
mouse_mean_d = pd.DataFrame()
for i in all_files:
    print('load file: ' + i)
    mouse_id.append(i[i.find('/',23)+1:i.find('/',25)])
    strain.append(i[i.find('/',20)+1:i.find('/',23)])
    group_id.append(i[i.find('/',17)+1:i.find('/',20)])
    stage.append(i[i.find('/',25)+1:i.find('/',32)])
    
    matfile = sio.loadmat(i, squeeze_me = True)
    eMouse = matfile['eMouse']
    # get locomotion trajectory
    mouse2D = mf.mouse2D_format(eMouse)
    walkTimes = mouse2D['walkTimes']
    params = mouse2D['params']
    pulseTimes = mouse2D['pulseTimes']
    
    # get kinematics
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
    if params['rec_duration']==900:
        camTimes = [0,900]
    else:
        camTimes = params['camTimes']
    bd_speed = kinematics['bd_speed'][int(np.round(camTimes[0]*params['fr'])):\
                                      int(np.round(camTimes[1]*params['fr']))]
    immobile_th = 2 # speed<2mm/s
    immobile_n = len(np.where(bd_speed<immobile_th)[0])
    immobile_t = immobile_n/params['fr']
    immobile_pct = immobile_t/(camTimes[1]-camTimes[0])*100
    
    # calculate the walking rate: number of walking bouts per minute
    walk_rate = kinematics['bd_start_stop'].shape[0]/(camTimes[1]-camTimes[0])*60 #second to min
    pre_walkTimes = kinematics['bd_start_stop'][np.where((kinematics['bd_start_stop'][:,1]<pulseTimes[0]) &\
                                       (kinematics['bd_start_stop'][:,0]>pulseTimes[0]-300))[0],:]
    post_walkTimes = kinematics['bd_start_stop'][np.where((kinematics['bd_start_stop'][:,0]>pulseTimes[0]) &\
                         (kinematics['bd_start_stop'][:,1]<pulseTimes[-1]))[0],:]
    back_walkTimes = kinematics['bd_start_stop'][np.where((kinematics['bd_start_stop'][:,0]>pulseTimes[-1]) &\
                         (kinematics['bd_start_stop'][:,1]<pulseTimes[-1]+300))[0],:]
        
    pre_walk_rate = pre_walkTimes.shape[0]/5
    post_walk_rate = post_walkTimes.shape[0]/5
    back_walk_rate = back_walkTimes.shape[0]/5
    
    # calculate the walking distance
    walk_distance = mf.walk_distance(mouse2D['body'], kinematics['bd_start_stop'], params)
    pre_walk_distance = mf.walk_distance(mouse2D['body'], pre_walkTimes, params)
    post_walk_distance = mf.walk_distance(mouse2D['body'], post_walkTimes, params)
    back_walk_distance = mf.walk_distance(mouse2D['body'], back_walkTimes, params)
    
    # calculate mean walk speed
    walk_speed = mf.walk_speed(kinematics['bd_speed'], kinematics['bd_start_stop'], params)
    pre_walk_speed = mf.walk_speed(kinematics['bd_speed'], pre_walkTimes, params)
    post_walk_speed = mf.walk_speed(kinematics['bd_speed'], post_walkTimes, params)
    back_walk_speed = mf.walk_speed(kinematics['bd_speed'], back_walkTimes, params)
    
    # calculate pre-post head angle change
    pre_head_angles, _ = mf.walk_angle_change(nose_tail_v,pre_walkTimes,params)
    post_head_angles, _ = mf.walk_angle_change(nose_tail_v,post_walkTimes,params)
    back_head_angles, _ = mf.walk_angle_change(nose_tail_v,back_walkTimes,params)
    
    # calculate pre-laser-post gaits
    pre_w_id = np.where((walkTimes[:,1]<pulseTimes[0]) &\
                        (walkTimes[:,0]>pulseTimes[0]-300))[0]
    post_w_id = np.where((walkTimes[:,0]>pulseTimes[0]) &\
                         (walkTimes[:,1]<pulseTimes[-1]))[0]
    back_w_id = np.where((walkTimes[:,0]>pulseTimes[-1]) &\
                         (walkTimes[:,1]<pulseTimes[-1]+300))[0]
    lf_st_id = np.where((stride_trimed['lf']['stride_flag']>=pre_w_id[0])&\
                        (stride_trimed['lf']['stride_flag']<=pre_w_id[-1]))[0]
    lr_st_id = np.where((stride_trimed['lr']['stride_flag']>=pre_w_id[0])&\
                        (stride_trimed['lr']['stride_flag']<=pre_w_id[-1]))[0]
    rf_st_id = np.where((stride_trimed['rf']['stride_flag']>=pre_w_id[0])&\
                        (stride_trimed['rf']['stride_flag']<=pre_w_id[-1]))[0]
    rr_st_id = np.where((stride_trimed['rr']['stride_flag']>=pre_w_id[0])&\
                        (stride_trimed['rr']['stride_flag']<=pre_w_id[-1]))[0]
    pre_gait = {
        'lf': gait_trimed['lf'].iloc[lf_st_id, :],
        'lr': gait_trimed['lr'].iloc[lr_st_id, :],
        'rf': gait_trimed['rf'].iloc[rf_st_id, :],
        'rr': gait_trimed['rr'].iloc[rr_st_id, :]}
    lf_st_id = np.where((stride_trimed['lf']['stride_flag']>=post_w_id[0])&\
                        (stride_trimed['lf']['stride_flag']<=post_w_id[-1]))[0]
    lr_st_id = np.where((stride_trimed['lr']['stride_flag']>=post_w_id[0])&\
                        (stride_trimed['lr']['stride_flag']<=post_w_id[-1]))[0]
    rf_st_id = np.where((stride_trimed['rf']['stride_flag']>=post_w_id[0])&\
                        (stride_trimed['rf']['stride_flag']<=post_w_id[-1]))[0]
    rr_st_id = np.where((stride_trimed['rr']['stride_flag']>=post_w_id[0])&\
                        (stride_trimed['rr']['stride_flag']<=post_w_id[-1]))[0]
    laser_gait = {
        'lf': gait_trimed['lf'].iloc[lf_st_id, :],
        'lr': gait_trimed['lr'].iloc[lr_st_id, :],
        'rf': gait_trimed['rf'].iloc[rf_st_id, :],
        'rr': gait_trimed['rr'].iloc[rr_st_id, :]}
    lf_st_id = np.where((stride_trimed['lf']['stride_flag']>=back_w_id[0])&\
                        (stride_trimed['lf']['stride_flag']<=back_w_id[-1]))[0]
    lr_st_id = np.where((stride_trimed['lr']['stride_flag']>=back_w_id[0])&\
                        (stride_trimed['lr']['stride_flag']<=back_w_id[-1]))[0]
    rf_st_id = np.where((stride_trimed['rf']['stride_flag']>=back_w_id[0])&\
                        (stride_trimed['rf']['stride_flag']<=back_w_id[-1]))[0]
    rr_st_id = np.where((stride_trimed['rr']['stride_flag']>=back_w_id[0])&\
                        (stride_trimed['rr']['stride_flag']<=back_w_id[-1]))[0]
    post_gait = {
        'lf': gait_trimed['lf'].iloc[lf_st_id, :],
        'lr': gait_trimed['lr'].iloc[lr_st_id, :],
        'rf': gait_trimed['rf'].iloc[rf_st_id, :],
        'rr': gait_trimed['rr'].iloc[rr_st_id, :]}
    
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
        'pre_head_angle': np.nanmean(pre_head_angles),
        'post_head_angle': np.nanmean(post_head_angles),
        'back_head_angle': np.nanmean(back_head_angles),
        'immobile_pct': immobile_pct,
        'walk_rate': walk_rate,
        'pre_walk_rate': pre_walk_rate,
        'post_walk_rate': post_walk_rate,
        'back_walk_rate': back_walk_rate,
        'walk_distance': np.sum(walk_distance),
        'pre_walk_distance':np.sum(pre_walk_distance),
        'post_walk_distance': np.sum(post_walk_distance),
        'back_walk_distance': np.sum(back_walk_distance),
        'walk_speed': np.mean(walk_speed),
        'pre_walk_speed': np.mean(pre_walk_speed),
        'post_walk_speed': np.mean(post_walk_speed),
        'back_walk_speed': np.mean(back_walk_speed),
        'stride_speed_lf_pre': np.mean(pre_gait['lf']['stride_velocity']),
        'stride_speed_lf_laser': np.mean(laser_gait['lf']['stride_velocity']),
        'stride_speed_lf_post': np.mean(post_gait['lf']['stride_velocity']),
        'stride_speed_lr_pre': np.mean(pre_gait['lr']['stride_velocity']),
        'stride_speed_lr_laser': np.mean(laser_gait['lr']['stride_velocity']),
        'stride_speed_lr_post': np.mean(post_gait['lr']['stride_velocity']),
        'stride_speed_rf_pre': np.mean(pre_gait['rf']['stride_velocity']),
        'stride_speed_rf_laser': np.mean(laser_gait['rf']['stride_velocity']),
        'stride_speed_rf_post': np.mean(post_gait['rf']['stride_velocity']),
        'stride_speed_rr_pre': np.mean(pre_gait['rr']['stride_velocity']),
        'stride_speed_rr_laser': np.mean(laser_gait['rr']['stride_velocity']),
        'stride_speed_rr_post': np.mean(post_gait['rr']['stride_velocity']),
        'stride_length_lf_pre': np.mean(pre_gait['lf']['stride_length']),
        'stride_length_lf_laser': np.mean(laser_gait['lf']['stride_length']),
        'stride_length_lf_post': np.mean(post_gait['lf']['stride_length']),
        'stride_length_lr_pre': np.mean(pre_gait['lr']['stride_length']),
        'stride_length_lr_laser': np.mean(laser_gait['lr']['stride_length']),
        'stride_length_lr_post': np.mean(post_gait['lr']['stride_length']),
        'stride_length_rf_pre': np.mean(pre_gait['rf']['stride_length']),
        'stride_length_rf_laser': np.mean(laser_gait['rf']['stride_length']),
        'stride_length_rf_post': np.mean(post_gait['rf']['stride_length']),
        'stride_length_rr_pre': np.mean(pre_gait['rr']['stride_length']),
        'stride_length_rr_laser': np.mean(laser_gait['rr']['stride_length']),
        'stride_length_rr_post': np.mean(post_gait['rr']['stride_length']),
        'stride_dur_lf_pre': np.mean(pre_gait['lf']['swing_duration']+\
                                 pre_gait['lf']['stance_duration']),
        'stride_dur_lf_laser': np.mean(laser_gait['lf']['swing_duration']+\
                                 laser_gait['lf']['stance_duration']),
        'stride_dur_lf_post': np.mean(post_gait['lf']['swing_duration']+\
                                 post_gait['lf']['stance_duration']),
        'stride_dur_lr_pre': np.mean(pre_gait['lr']['swing_duration']+\
                                 pre_gait['lr']['stance_duration']),
        'stride_dur_lr_laser': np.mean(laser_gait['lr']['swing_duration']+\
                                 laser_gait['lr']['stance_duration']),
        'stride_dur_lr_post': np.mean(post_gait['lr']['swing_duration']+\
                                 post_gait['lr']['stance_duration']),
        'stride_dur_rf_pre': np.mean(pre_gait['rf']['swing_duration']+\
                                 pre_gait['rf']['stance_duration']),
        'stride_dur_rf_laser': np.mean(laser_gait['rf']['swing_duration']+\
                                 laser_gait['rf']['stance_duration']),
        'stride_dur_rf_post': np.mean(post_gait['rf']['swing_duration']+\
                                 post_gait['rf']['stance_duration']),
        'stride_dur_rr_pre': np.mean(pre_gait['rr']['swing_duration']+\
                                 pre_gait['rr']['stance_duration']),
        'stride_dur_rr_laser': np.mean(laser_gait['rr']['swing_duration']+\
                                 laser_gait['rr']['stance_duration']),
        'stride_dur_rr_post': np.mean(post_gait['rr']['swing_duration']+\
                                 post_gait['rr']['stance_duration']),
        'sw_st_ratio_lf_pre': np.mean(pre_gait['lf']['swing_stance_ratio']),
        'sw_st_ratio_lf_laser': np.mean(laser_gait['lf']['swing_stance_ratio']),
        'sw_st_ratio_lf_post': np.mean(post_gait['lf']['swing_stance_ratio']),
        'sw_st_ratio_lr_pre': np.mean(pre_gait['lr']['swing_stance_ratio']),
        'sw_st_ratio_lr_laser': np.mean(laser_gait['lr']['swing_stance_ratio']),
        'sw_st_ratio_lr_post': np.mean(post_gait['lr']['swing_stance_ratio']),
        'sw_st_ratio_rf_pre': np.mean(pre_gait['rf']['swing_stance_ratio']),
        'sw_st_ratio_rf_laser': np.mean(laser_gait['rf']['swing_stance_ratio']),
        'sw_st_ratio_rf_post': np.mean(post_gait['rf']['swing_stance_ratio']),
        'sw_st_ratio_rr_pre': np.mean(pre_gait['rr']['swing_stance_ratio']),
        'sw_st_ratio_rr_laser': np.mean(laser_gait['rr']['swing_stance_ratio']),
        'sw_st_ratio_rr_post': np.mean(post_gait['rr']['swing_stance_ratio']),
        'stride_speed_cv_lf_pre': variation(pre_gait['lf']['stride_velocity']),
        'stride_speed_cv_lf_laser': variation(laser_gait['lf']['stride_velocity']),
        'stride_speed_cv_lf_post': variation(post_gait['lf']['stride_velocity']),
        'stride_speed_cv_lr_pre': variation(pre_gait['lr']['stride_velocity']),
        'stride_speed_cv_lr_laser': variation(laser_gait['lr']['stride_velocity']),
        'stride_speed_cv_lr_post': variation(post_gait['lr']['stride_velocity']),
        'stride_speed_cv_rf_pre': variation(pre_gait['rf']['stride_velocity']),
        'stride_speed_cv_rf_laser': variation(laser_gait['rf']['stride_velocity']),
        'stride_speed_cv_rf_post': variation(post_gait['rf']['stride_velocity']),
        'stride_speed_cv_rr_pre': variation(pre_gait['rr']['stride_velocity']),
        'stride_speed_cv_rr_laser': variation(laser_gait['rr']['stride_velocity']),
        'stride_speed_cv_rr_post': variation(post_gait['rr']['stride_velocity']),
        'stride_length_cv_lf_pre': variation(pre_gait['lf']['stride_length']),
        'stride_length_cv_lf_laser': variation(laser_gait['lf']['stride_length']),
        'stride_length_cv_lf_post': variation(post_gait['lf']['stride_length']),
        'stride_length_cv_lr_pre': variation(pre_gait['lr']['stride_length']),
        'stride_length_cv_lr_laser': variation(laser_gait['lr']['stride_length']),
        'stride_length_cv_lr_post': variation(post_gait['lr']['stride_length']),
        'stride_length_cv_rf_pre': variation(pre_gait['rf']['stride_length']),
        'stride_length_cv_rf_laser': variation(laser_gait['rf']['stride_length']),
        'stride_length_cv_rf_post': variation(post_gait['rf']['stride_length']),
        'stride_length_cv_rr_pre': variation(pre_gait['rr']['stride_length']),
        'stride_length_cv_rr_laser': variation(laser_gait['rr']['stride_length']),
        'stride_length_cv_rr_post': variation(post_gait['rr']['stride_length']),
        'stride_dur_cv_lf_pre': variation(pre_gait['lf']['swing_duration']+\
                                 pre_gait['lf']['stance_duration']),
        'stride_dur_cv_lf_laser': variation(laser_gait['lf']['swing_duration']+\
                                 laser_gait['lf']['stance_duration']),
        'stride_dur_cv_lf_post': variation(post_gait['lf']['swing_duration']+\
                                 post_gait['lf']['stance_duration']),
        'stride_dur_cv_lr_pre': variation(pre_gait['lr']['swing_duration']+\
                                 pre_gait['lr']['stance_duration']),
        'stride_dur_cv_lr_laser': variation(laser_gait['lr']['swing_duration']+\
                                 laser_gait['lr']['stance_duration']),
        'stride_dur_cv_lr_post': variation(post_gait['lr']['swing_duration']+\
                                 post_gait['lr']['stance_duration']),
        'stride_dur_cv_rf_pre': variation(pre_gait['rf']['swing_duration']+\
                                 pre_gait['rf']['stance_duration']),
        'stride_dur_cv_rf_laser': variation(laser_gait['rf']['swing_duration']+\
                                 laser_gait['rf']['stance_duration']),
        'stride_dur_cv_rf_post': variation(post_gait['rf']['swing_duration']+\
                                 post_gait['rf']['stance_duration']),
        'stride_dur_cv_rr_pre': variation(pre_gait['rr']['swing_duration']+\
                                 pre_gait['rr']['stance_duration']),
        'stride_dur_cv_rr_laser': variation(laser_gait['rr']['swing_duration']+\
                                 laser_gait['rr']['stance_duration']),
        'stride_dur_cv_rr_post': variation(post_gait['rr']['swing_duration']+\
                                 post_gait['rr']['stance_duration']),
        'sw_st_ratio_cv_lf_pre': variation(pre_gait['lf']['swing_stance_ratio']),
        'sw_st_ratio_cv_lf_laser': variation(laser_gait['lf']['swing_stance_ratio']),
        'sw_st_ratio_cv_lf_post': variation(post_gait['lf']['swing_stance_ratio']),
        'sw_st_ratio_cv_lr_pre': variation(pre_gait['lr']['swing_stance_ratio']),
        'sw_st_ratio_cv_lr_laser': variation(laser_gait['lr']['swing_stance_ratio']),
        'sw_st_ratio_cv_lr_post': variation(post_gait['lr']['swing_stance_ratio']),
        'sw_st_ratio_cv_rf_pre': variation(pre_gait['rf']['swing_stance_ratio']),
        'sw_st_ratio_cv_rf_laser': variation(laser_gait['rf']['swing_stance_ratio']),
        'sw_st_ratio_cv_rf_post': variation(post_gait['rf']['swing_stance_ratio']),
        'sw_st_ratio_cv_rr_pre': variation(pre_gait['rr']['swing_stance_ratio']),
        'sw_st_ratio_cv_rr_laser': variation(laser_gait['rr']['swing_stance_ratio']),
        'sw_st_ratio_cv_rr_post': variation(post_gait['rr']['swing_stance_ratio']),        
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

'''
######################################
create path for storage
######################################
'''
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']
output_path = all_files[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
'''
######################################
save data
######################################
'''
print('data saved to: ' + output_path)

with open(output_path + 'all-single-mice-gaits-laser-stim.pickle', 'wb') as f:
    pickle.dump(mouse_mean_d, f)
#Save gaits data into excel
mouse_mean_d.to_csv(output_path + 'all-single-mice-gaits-laser-stim.csv')