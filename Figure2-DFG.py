# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:12:54 2023
Figure 2. striatal neurons are phase locked to gait cycle

@author: lonya
"""
#%% Load Data
from tkinter import messagebox
import tkinter as tk
import tkinter.filedialog as fd
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
import scipy.io as sio
import my_funcs as mf

#%%
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Select M2216.mat or M2210.mat files',\
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
#%% processing data
mouse_id = all_files[0][len(all_files[0])-9:-4]
mouse_gait = pd.DataFrame()
    
matfile = sio.loadmat(all_files[0], squeeze_me = True)
eMouse = matfile['eMouse']
# get locomotion trajectory
mouse2D = mf.mouse2D_format(eMouse)
walkTimes = mouse2D['walkTimes']
params = mouse2D['params']

# calculate self-view mouse, gait cycle
svm = mf.createSVM(mouse2D)

# get phase angle
svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')

# get kinematics
kinematics = mf.mouse_kinematics(mouse2D, params)

# get ephys data
st_mtx, camTimes, tagging, cln, clwf, fs, xcoords, ycoords, unitBestCh = \
    mf.ephys_format(eMouse)

# get gait info
#gait, stride = calculate_gait(svm, mouse2D)
#svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')
gait, stride = mf.calculate_gait(svm_bf, mouse2D) #1/11/2023
gait_trimed, stride_trimed = mf.trim_gait(gait, stride)

params['n_circ_bins'] = 24
params['density'] = False
params['norm']=True

# calculate limb phase angle
svm_angle = mf.get_mouse_angle_pct(svm_bf, stride_trimed)
#svm_angle = mf.get_mouse_angle(svm_bf)

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

#%% create path for figures
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']

output_path = all_files[0][:backslash_id[-1]+1]

figpath = str(output_path) + 'Figure 2/'

if not os.path.exists(figpath):
    os.makedirs(figpath)

#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42  

#%% plot firing rate in time&phase domain of 2 example neurons - Figure 2DFG
'''
M2216 - cluster_id=8
M2210 - cluster_id=2
'''
if mouse_id=='M2216':
    cluster_id = 8
elif mouse_id=='M2210':
    cluster_id = 2
else:
    cluster_id=0
    print('please input your own cluster ID')
    
limb = 'lr'
# get strance time points, sorted by the stride duration
event1 = stride_trimed[limb]['stride'][:,0]
stride_dur = stride_trimed[limb]['stride'][:,2]-stride_trimed[limb]['stride'][:,0]
stance_time = stride_trimed[limb]['stride'][:,[0,2]]
event2_stances = stance_time - stride_trimed[limb]['stride'][:,[0,0]]
sort_id = np.argsort(-1*stride_dur)
window = [-0.5, 0.55] # in second
binsize = 0.025 # 25ms
sm_n=3

# 1. get spikes raster & mean firing rate in time domain
sp = st_mtx[cluster_id]
event1_spike_times = mf.restrict_spike_times(st = sp, event = event1, win=window)
t, psth1, psth1_sem,_ = \
    mf.bins_spike_times(spike_times = event1_spike_times, \
                        bs = binsize, win = window)
psth1_sm = gaussian_filter(psth1, sigma = sm_n)
# save data for graphpad prism
t_fr = zip(t.tolist(), psth1_sm.tolist(), psth1_sem.tolist())
t_fr_df = pd.DataFrame(t_fr, columns=['time', 'firing rate', 'sem'])
#t_fr_df.to_csv(f'{output}mean_firingrate2time_{mouse_id[-1]}_{cluster_id}_{limb}.csv')
    
# 2. get firing rate in phase domain (2 cycles for illustration)
theta1 = np.linspace(0,720,params['n_circ_bins']*2) #for linear plot, 2 cycles
theta2 = np.linspace(0,360,params['n_circ_bins'],endpoint=False) # for polar plot
theta2 = np.append(theta2, [theta2[0]],axis=0)
# get 2 cycles radii
radii1 = np.concatenate((svm_sp_hmtx_str[limb][cluster_id,:], \
                      svm_sp_hmtx_str[limb][cluster_id,:]),axis=0)
radii_sm1 = gaussian_filter(radii1, sigma = 2, mode='wrap')
# save data for graphpad prism
p_fr = zip(theta1.tolist(), radii_sm1.tolist())
p_fr_df = pd.DataFrame(p_fr, columns=['phase', 'firing rate'])
#p_fr_df.to_csv(f'{output}mean_firingrate2phase_{mouse_id[-1]}_{cluster_id}_{limb}.csv')

# 3. get polar plot data
radii_lr = np.concatenate((svm_sp_hmtx_str['lr'][cluster_id,:], \
                        svm_sp_hmtx_str['lr'][cluster_id,:], \
                        svm_sp_hmtx_str['lr'][cluster_id,:]),axis=0)
radii_lr_sm = gaussian_filter(radii_lr, sigma = 2, mode='wrap')
radii_lr_sm = radii_lr_sm[24:49] # get 1 cycle for polar plot
# get mean vector length
r_lr = svm_spike_r_str['lr'][cluster_id,:]
# scale radii distribution, so maximum equal to mean vector length
radii_scaler = max(radii_lr_sm)/r_lr[0]
radii_lr_sm_norm = radii_lr_sm/radii_scaler

# plot figures
fig = plt.figure(figsize=(12, 12))
ax0 = fig.add_subplot(221)

ax0.eventplot(event1_spike_times[sort_id],color='k',linelengths=3,linewidths=1)
ax0.eventplot(event2_stances[sort_id,:],color='r',linelengths=3, linewidths=1)
ax0.set_xlim(window)
ax0.spines[['top', 'right']].set_visible(False)
ax0.set_xlabel("stance start (s)", fontsize=24, family='arial')
ax0.set_ylabel("strides", fontsize=24, family='arial')
ax0.tick_params(direction='out', width=1, length=5, labelsize=24)

ax1 = fig.add_subplot(222)
ax1.plot(theta1,radii_sm1, color = 'r', lw=1)
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_xlim([0,270])
ax1.set_xticks([0,180,360,540,720])
#ax1.set_ylim([8, 16])
ax1.set_xlabel('limb phase (deg)', fontsize=24, family='arial')
ax1.set_ylabel('firing rate (Hz)', fontsize=24, family='arial')
ax1.tick_params(direction='out', width=1, length=5, labelsize=24)

ax2 = fig.add_subplot(223)
ax2.set_xlim([-0.5,0.5])
#ax2.set_ylim([9.5,16])
ax2.plot(t,psth1_sm, color = 'k', lw=1)
ax2.spines[['top', 'right']].set_visible(False)
ax2.fill_between(t, psth1_sm+psth1_sem,\
                 psth1_sm-psth1_sem, \
                 color = 'k', alpha=.3, linewidth=0)
ax2.set_xlim(window)
ax2.set_xlabel("stance start (s)", fontsize=24, family='arial')
ax2.set_ylabel("firing rate (Hz)", fontsize=24, family='arial')
ax2.tick_params(direction='out', width=1, length=5, labelsize=24)

ax3 = fig.add_subplot(224, projection='polar')
ax3.plot(np.deg2rad(theta2),radii_lr_sm_norm, color = '#00FF00', lw=2)
ax3.plot([0,r_lr[1]],[0,r_lr[0]],linewidth=2,color='#00FF00')

ax3.set_rgrids([0.1, 0.2], labels=None, angle=45, fontsize=24, family='arial')
ax3.set_thetagrids(range(0, 360, 45), ('0','', '90','', '180','', '270',''), \
                   fontsize=24, family='arial')
plt.tight_layout()

if mouse_id == 'M2216':
    plt.savefig(f'{figpath}Fig2-DEG-right.pdf', dpi=300, bbox_inches='tight', transparent=True)
elif mouse_id=='2210':
    plt.savefig(f'{figpath}Fig1-DEG-left.pdf', dpi=300, bbox_inches='tight', transparent=True)
else:
    plt.savefig(f'{figpath}Fig1-DEG-{mouse_id}.pdf', dpi=300, bbox_inches='tight', transparent=True)