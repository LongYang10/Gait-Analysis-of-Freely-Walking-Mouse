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

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import numpy as np
from scipy.stats import variation 
import seaborn as sns
#%% Load Data (all laser manipulation animal)

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

#%% Load previous saved data path
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, title='Choose a file')
root.destroy()
print(filez)

with open(filez[0], 'rb') as f:
    all_files = pickle.load(f)
#%% save data path for future use

outpath = all_files[0][:all_files[0].find('/',10)+1]+'output/'
print('data saved to: ' + outpath)

#Save gaits data into pickle
with open(outpath + 'all_mice_datapath_laser_stim.pickle', 'wb') as f:
    pickle.dump(all_files, f)


#%% processing data

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
    
#%% t test for pre/post laser

# remove mouse M2338
mouse_mean_d = mouse_mean_d.drop(index=3)
mouse_select = mouse_mean_d.loc[(mouse_mean_d['group_id']=='LA')]

new_mouse_select = {
    'group_id': mouse_select['group_id'].values.tolist()+\
                mouse_select['group_id'].values.tolist()+\
                mouse_select['group_id'].values.tolist(),
    'laser':["pre" for x in range(len(mouse_select.index))]+\
            ["laser" for x in range(len(mouse_select.index))]+\
            ["post" for x in range(len(mouse_select.index))],
    'walk_rate': mouse_select['pre_walk_rate'].values.tolist()+\
                 mouse_select['post_walk_rate'].values.tolist()+\
                 mouse_select['back_walk_rate'].values.tolist(),
    'walk_distance':mouse_select['pre_walk_distance'].values.tolist()+\
                    mouse_select['post_walk_distance'].values.tolist()+\
                    mouse_select['back_walk_distance'].values.tolist(),
    'head_angle': mouse_select['pre_head_angle'].values.tolist()+\
                  mouse_select['post_head_angle'].values.tolist()+\
                  mouse_select['back_head_angle'].values.tolist(),
    'walk_speed':mouse_select['pre_walk_speed'].values.tolist()+\
                 mouse_select['post_walk_speed'].values.tolist()+\
                 mouse_select['back_walk_speed'].values.tolist()
                  }
mouse_select_new = pd.DataFrame(data=new_mouse_select)

#%% comparison pre/laser/post, #Figure 6 – Figure Supplement 2B,C,D,E
from scipy.stats import ttest_rel
laser_pre_post_walk_rate = ttest_rel(mouse_select['pre_walk_rate'],\
                                mouse_select['post_walk_rate']).pvalue
    
laser_pre_post_distance = ttest_rel(mouse_select['pre_walk_distance'],\
                                mouse_select['post_walk_distance']).pvalue
laser_pre_post_walk_speed= ttest_rel(mouse_select['pre_walk_speed'],\
                                mouse_select['post_walk_speed']).pvalue

fig = plt.figure(figsize=(8,12))
### mean heading comparison pre/laser/post, #Figure 6 – Figure Supplement 2B
ax1 = fig.add_subplot(221)
sns.pointplot(data=mouse_select_new, x="laser", y="head_angle", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="sd", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
ax1.set_ylabel('mean change in heading (deg)', fontsize=24, family='arial')
ax1.set(xlabel=None)
#ax1.set_ylim([-60,20])
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=24)
ax1.set_xticklabels(["pre", "laser", "post"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

# significance test
hd_a_ct = mouse_select['pre_head_angle'].values
hd_a_pd = mouse_select['post_head_angle'].values
hd_a_bk = mouse_select['back_head_angle'].values
hd_m_ct, hd_sd_ct,_ = mf.circ_m(hd_a_ct)
hd_m_pd, hd_sd_pd,_ = mf.circ_m(hd_a_pd)
hd_m_bk, hd_sd_bk,_ = mf.circ_m(hd_a_bk)
#hd_p = mf.permut_angle(hd_a_ct, hd_a_pd, 1000)
# save data for graphpad prism
hd_a = [[hd_m_ct, hd_sd_ct, hd_m_pd, hd_sd_pd, hd_m_bk, hd_sd_bk]]
hd_a_df = pd.DataFrame(hd_a,columns=['pre-mean','pre-sem',\
                                     'post-mean','post-sem',\
                                     'back-mean', 'back_sem'])
#hd_a_df.to_csv(f'{output}head_angle_CT_PD.csv')

### mean walking speed comparison pre/laser/post, #Figure 6 – Figure Supplement 2C
ax2 = fig.add_subplot(222)
sns.pointplot(data=mouse_select_new, x="laser", y="walk_speed", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylabel('walking body speed (mm/s)', fontsize=24, family='arial')
ax2.set(xlabel=None)
#ax2.set_ylim([0,200])
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=24)
ax2.set_xticklabels(["pre", "laser", "post"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')
ax2.set_title(f'p = {np.round(laser_pre_post_walk_speed,3)}')

### total travel distance comparison pre/laser/post, #Figure 6 – Figure Supplement 2D
ax3 = fig.add_subplot(223)
sns.pointplot(data=mouse_select_new, x="laser", y="walk_distance", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax3)
ax3.set_ylabel('total distance traveled (mm)', fontsize=24, family='arial')
ax3.set(xlabel=None)
#ax3.set_ylim([0,100000])
ax3.spines[['left','bottom']].set_linewidth(2)
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(direction='out', width=2, length=8, labelsize=24)
ax3.set_xticklabels(["pre", "laser", "post"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(ax3.get_yticks(),family='arial')
ax3.set_title(f'p = {np.round(laser_pre_post_distance,3)}')

### initiation rate comparison pre/laser/post, #Figure 6 – Figure Supplement 2E
ax4 = fig.add_subplot(224)
sns.pointplot(data=mouse_select_new, x="laser", y="walk_rate", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax4)
ax4.set_ylabel('initiation rate (starts/min)', fontsize=24, family='arial')
ax4.set(xlabel=None)
#ax4.set_ylim([0,15])
ax4.spines[['left','bottom']].set_linewidth(2)
ax4.spines[['top', 'right']].set_visible(False)
ax4.tick_params(direction='out', width=2, length=8, labelsize=24)
ax4.set_xticklabels(["pre", "laser", "post"],rotation=45)
#ax4.set_yticks([0.0, 0.2, 0.4])
ax4.set_yticklabels(ax4.get_yticks(),family='arial')
ax4.set_title(f'p = {np.round(laser_pre_post_walk_rate,3)}')

plt.tight_layout()

#%% mean gaits comparison pre-laser vs laser
#Figure 6 – Figure Supplement 2F,G,H,I
#prepare dataframe 
new_limb_select = {
    'laser':["pre" for x in range(len(mouse_select.index))]+\
            ["laser" for x in range(len(mouse_select.index))]+\
            ["post" for x in range(len(mouse_select.index))]+\
            ["pre" for x in range(len(mouse_select.index))]+\
            ["laser" for x in range(len(mouse_select.index))]+\
            ["post" for x in range(len(mouse_select.index))]+\
            ["pre" for x in range(len(mouse_select.index))]+\
            ["laser" for x in range(len(mouse_select.index))]+\
            ["post" for x in range(len(mouse_select.index))]+\
            ["pre" for x in range(len(mouse_select.index))]+\
            ["laser" for x in range(len(mouse_select.index))]+\
            ["post" for x in range(len(mouse_select.index))],
    'limb':["lf" for x in range(len(mouse_select.index)*3)]+\
           ["lr" for x in range(len(mouse_select.index)*3)]+\
           ["rf" for x in range(len(mouse_select.index)*3)]+\
           ["rr" for x in range(len(mouse_select.index)*3)],
    'stride_length': mouse_select['stride_length_lf_pre'].values.tolist()+\
                     mouse_select['stride_length_lf_laser'].values.tolist()+\
                     mouse_select['stride_length_lf_post'].values.tolist()+\
                     mouse_select['stride_length_lr_pre'].values.tolist()+\
                     mouse_select['stride_length_lr_laser'].values.tolist()+\
                     mouse_select['stride_length_lr_post'].values.tolist()+\
                     mouse_select['stride_length_rf_pre'].values.tolist()+\
                     mouse_select['stride_length_rf_laser'].values.tolist()+\
                     mouse_select['stride_length_rf_post'].values.tolist()+\
                     mouse_select['stride_length_rr_pre'].values.tolist()+\
                     mouse_select['stride_length_rr_laser'].values.tolist()+\
                     mouse_select['stride_length_rr_post'].values.tolist(),
    'stride_dur':mouse_select['stride_dur_lf_pre'].values.tolist()+\
                mouse_select['stride_dur_lf_laser'].values.tolist()+\
                mouse_select['stride_dur_lf_post'].values.tolist()+\
                mouse_select['stride_dur_lr_pre'].values.tolist()+\
                mouse_select['stride_dur_lr_laser'].values.tolist()+\
                mouse_select['stride_dur_lr_post'].values.tolist()+\
                mouse_select['stride_dur_rf_pre'].values.tolist()+\
                mouse_select['stride_dur_rf_laser'].values.tolist()+\
                mouse_select['stride_dur_rf_post'].values.tolist()+\
                mouse_select['stride_dur_rr_pre'].values.tolist()+\
                mouse_select['stride_dur_rr_laser'].values.tolist()+\
                mouse_select['stride_dur_rr_post'].values.tolist(),
    'stride_speed': mouse_select['stride_speed_lf_pre'].values.tolist()+\
                    mouse_select['stride_speed_lf_laser'].values.tolist()+\
                    mouse_select['stride_speed_lf_post'].values.tolist()+\
                    mouse_select['stride_speed_lr_pre'].values.tolist()+\
                    mouse_select['stride_speed_lr_laser'].values.tolist()+\
                    mouse_select['stride_speed_lr_post'].values.tolist()+\
                    mouse_select['stride_speed_rf_pre'].values.tolist()+\
                    mouse_select['stride_speed_rf_laser'].values.tolist()+\
                    mouse_select['stride_speed_rf_post'].values.tolist()+\
                    mouse_select['stride_speed_rr_pre'].values.tolist()+\
                    mouse_select['stride_speed_rr_laser'].values.tolist()+\
                    mouse_select['stride_speed_rr_post'].values.tolist(),
    'sw_st_ratio': mouse_select['sw_st_ratio_lf_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_lf_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_lf_post'].values.tolist()+\
                    mouse_select['sw_st_ratio_lr_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_lr_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_lr_post'].values.tolist()+\
                    mouse_select['sw_st_ratio_rf_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_rf_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_rf_post'].values.tolist()+\
                    mouse_select['sw_st_ratio_rr_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_rr_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_rr_post'].values.tolist(),
    'stride_length_cv':mouse_select['stride_length_cv_lf_pre'].values.tolist()+\
                    mouse_select['stride_length_cv_lf_laser'].values.tolist()+\
                    mouse_select['stride_length_cv_lf_post'].values.tolist()+\
                    mouse_select['stride_length_cv_lr_pre'].values.tolist()+\
                    mouse_select['stride_length_cv_lr_laser'].values.tolist()+\
                    mouse_select['stride_length_cv_lr_post'].values.tolist()+\
                    mouse_select['stride_length_cv_rf_pre'].values.tolist()+\
                    mouse_select['stride_length_cv_rf_laser'].values.tolist()+\
                    mouse_select['stride_length_cv_rf_post'].values.tolist()+\
                    mouse_select['stride_length_cv_rr_pre'].values.tolist()+\
                    mouse_select['stride_length_cv_rr_laser'].values.tolist()+\
                    mouse_select['stride_length_cv_rr_post'].values.tolist(),
    'stride_dur_cv':mouse_select['stride_dur_cv_lf_pre'].values.tolist()+\
                    mouse_select['stride_dur_cv_lf_laser'].values.tolist()+\
                    mouse_select['stride_dur_cv_lf_post'].values.tolist()+\
                    mouse_select['stride_dur_cv_lr_pre'].values.tolist()+\
                    mouse_select['stride_dur_cv_lr_laser'].values.tolist()+\
                    mouse_select['stride_dur_cv_lr_post'].values.tolist()+\
                    mouse_select['stride_dur_cv_rf_pre'].values.tolist()+\
                    mouse_select['stride_dur_cv_rf_laser'].values.tolist()+\
                    mouse_select['stride_dur_cv_rf_post'].values.tolist()+\
                    mouse_select['stride_dur_cv_rr_pre'].values.tolist()+\
                    mouse_select['stride_dur_cv_rr_laser'].values.tolist()+\
                    mouse_select['stride_dur_cv_rr_post'].values.tolist(),
    'stride_speed_cv': mouse_select['stride_speed_cv_lf_pre'].values.tolist()+\
                    mouse_select['stride_speed_cv_lf_laser'].values.tolist()+\
                    mouse_select['stride_speed_cv_lf_post'].values.tolist()+\
                    mouse_select['stride_speed_cv_lr_pre'].values.tolist()+\
                    mouse_select['stride_speed_cv_lr_laser'].values.tolist()+\
                    mouse_select['stride_speed_cv_lr_post'].values.tolist()+\
                    mouse_select['stride_speed_cv_rf_pre'].values.tolist()+\
                    mouse_select['stride_speed_cv_rf_laser'].values.tolist()+\
                    mouse_select['stride_speed_cv_rf_post'].values.tolist()+\
                    mouse_select['stride_speed_cv_rr_pre'].values.tolist()+\
                    mouse_select['stride_speed_cv_rr_laser'].values.tolist()+\
                    mouse_select['stride_speed_cv_rr_post'].values.tolist(),
    'sw_st_ratio_cv':mouse_select['sw_st_ratio_cv_lf_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_lf_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_lf_post'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_lr_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_lr_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_lr_post'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_rf_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_rf_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_rf_post'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_rr_pre'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_rr_laser'].values.tolist()+\
                    mouse_select['sw_st_ratio_cv_rr_post'].values.tolist()
                  }
limb_select_df = pd.DataFrame(data=new_limb_select)

fig = plt.figure(figsize=(12,8))

### plot stride length - Figure 5G
ax1 = fig.add_subplot(221)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=limb_select_df, x="limb", y="stride_length", hue="laser", \
              palette=['#0000FF','#FF0000','#00FF00'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
ax1.set_ylim([45,65])
ax1.get_legend().remove()
ax1.set_ylabel('mean stride length (mm)', fontsize=16, family='arial')
ax1.set(xlabel=None)
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=16)
ax1.set_xticklabels(["LF", "LR", "RF", "RR"])
#ax1.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

### plot stride duration - Figure 5H
ax2 = fig.add_subplot(222)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=limb_select_df, x="limb", y="stride_dur", hue="laser", \
              palette=['#0000FF','#FF0000','#00FF00'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylim([0.25,0.4])
ax2.get_legend().remove()
ax2.set_ylabel('mean stride duration (s)', fontsize=16, family='arial')
ax2.set(xlabel=None)
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=16)
ax2.set_xticklabels(["LF", "LR", "RF", "RR"])
ax2.set_yticks([0.25, 0.3, 0.35, 0.4])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')

### plot stride speed - Figure 5I
ax3 = fig.add_subplot(223)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=limb_select_df, x="limb", y="stride_speed", hue="laser", \
              palette=['#0000FF','#FF0000','#00FF00'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax3)
ax3.set_ylim([120,200])
ax3.get_legend().remove()
ax3.set_ylabel('mean stride speed (mm/s)', fontsize=16, family='arial')
ax3.set(xlabel=None)
ax3.spines[['left','bottom']].set_linewidth(2)
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(direction='out', width=2, length=8, labelsize=16)
ax3.set_xticklabels(["LF", "LR", "RF", "RR"])
#ax3.set_yticks([0.25, 0.3, 0.35, 0.4, 0.45])
ax3.set_yticklabels(ax3.get_yticks(),family='arial')

### plot swing stance ratio - Figure 5J
ax4 = fig.add_subplot(224)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=limb_select_df, x="limb", y="sw_st_ratio", hue="laser", \
              palette=['#0000FF','#FF0000','#00FF00'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax4)
ax4.set_ylim([0.5,1.0])
ax4.get_legend().remove()
ax4.set_ylabel('mean swing:stance duration', fontsize=16, family='arial')
ax4.set(xlabel=None)
ax4.spines[['left','bottom']].set_linewidth(2)
ax4.spines[['top', 'right']].set_visible(False)
ax4.tick_params(direction='out', width=2, length=8, labelsize=16)
ax4.set_xticklabels(["LF", "LR", "RF", "RR"])
ax4.set_yticks([0.6,0.8])
ax4.set_yticklabels(ax4.get_yticks(),family='arial')
ax4.legend()

#plt.savefig(f'{figpath}Figure5-GHIJ.pdf',dpi=300,bbox_inches='tight', transparent=True)
#%% gait CV comparision sham vs 6OHDA  - Figure S2-CDEF
fig = plt.figure(figsize=(12,8))

### plot stride length CV- Figure S2C
ax1 = fig.add_subplot(221)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=mouse_select_new, x="limb", y="stride_length_cv", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
ax1.set_ylim([0.15,0.3])
ax1.get_legend().remove()
ax1.set_ylabel('stride length CV (mm)', fontsize=16, family='arial')
ax1.set(xlabel=None)
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=16)
ax1.set_xticklabels(["LF", "LR", "RF", "RR"])
ax1.set_yticks([0.15, 0.20, 0.25, 0.30])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

### plot stride duration CV - Figure S2D
ax2 = fig.add_subplot(222)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=mouse_select_new, x="limb", y="stride_dur_cv", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylim([0.16,0.21])
ax2.get_legend().remove()
ax2.set_ylabel('stride duration CV (s)', fontsize=16, family='arial')
ax2.set(xlabel=None)
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=16)
ax2.set_xticklabels(["LF", "LR", "RF", "RR"])
ax2.set_yticks([0.16, 0.18, 0.2])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')

### plot stride speed CV - Figure S2E
ax3 = fig.add_subplot(223)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=mouse_select_new, x="limb", y="stride_speed_cv", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax3)
ax3.set_ylim([0.24,0.34])
ax3.get_legend().remove()
ax3.set_ylabel('stride speed CV (mm/s)', fontsize=16, family='arial')
ax3.set(xlabel=None)
ax3.spines[['left','bottom']].set_linewidth(2)
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(direction='out', width=2, length=8, labelsize=16)
ax3.set_xticklabels(["LF", "LR", "RF", "RR"])
ax3.set_yticks([0.24, 0.26, 0.28, 0.3, 0.32,0.34])
ax3.set_yticklabels(ax3.get_yticks(),family='arial')

### plot swing stance ratio CV - Figure S2F
ax4 = fig.add_subplot(224)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=mouse_select_new, x="limb", y="sw_st_ratio_cv", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax4)
ax4.set_ylim([0.0,0.6])
ax4.get_legend().remove()
ax4.set_ylabel('swing:stance duration CV', fontsize=16, family='arial')
ax4.set(xlabel=None)
ax4.spines[['left','bottom']].set_linewidth(2)
ax4.spines[['top', 'right']].set_visible(False)
ax4.tick_params(direction='out', width=2, length=8, labelsize=16)
ax4.set_xticklabels(["LF", "LR", "RF", "RR"])
ax4.set_yticks([0.0,0.2,0.4,0.6])
ax4.set_yticklabels(ax4.get_yticks(),family='arial')

#plt.savefig(f'{figpath}Figure S2-CDEF.pdf',dpi=300,bbox_inches='tight', transparent=True)