# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:11:06 2023
Figure 3. Dorsal striatal neurons encoding single-limb phase angle, body speed,
 and start/stop of walking.
 
@author: lonya
"""
#%% Load Data
from tkinter import messagebox
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import os
#%% Select eMouse file
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Select M2216.mat or F2208.mat files',\
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

#%% create path for figures
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']

output_path = all_files[0][:backslash_id[-1]+1]

figpath = str(output_path) + 'Figure 3/'

if not os.path.exists(figpath):
    os.makedirs(figpath)
    
#%% processing data
print('load file: ' + all_files[0])
mouse_id = all_files[0][len(all_files[0])-9:-4]

matfile = sio.loadmat(all_files[0], squeeze_me = True)
eMouse = matfile['eMouse']
# get locomotion trajectory
mouse2D = mf.mouse2D_format(eMouse)
walkTimes = mouse2D['walkTimes']
params = mouse2D['params']

# calculate self-view mouse, gait cycle
svm = mf.createSVM(mouse2D)

# get filtered self-view mouse
svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')

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
params['minimum_speed'] = 0   # 0 for plots; 50mm/s for speed score
params['window'] = [0, params['rec_duration']]
params['remove_bad_bins'] = False
   
binning_speed, binning_fr, binning_se = \
    mf.calculate_speed_score(kinematics['bd_speed'],st_mtx, params)
rval, pval = np.zeros(binning_fr.shape[1]), np.zeros(binning_fr.shape[1])
for i in range(binning_fr.shape[1]):
    rval[i],pval[i] = pearsonr(binning_speed, binning_fr[:,i])

#calculate the firing rate around start/stop of all neurons
t, psth_start, start_se, start_p,_  = \
    mf.psth_norm(kinematics['bd_start_stop'][:,0], \
                 st_mtx, np.ones(len(st_mtx)), params) #start times
_, psth_stop, stop_se, stop_p,_ = \
    mf.psth_norm(kinematics['bd_start_stop'][:,1], \
                 st_mtx, np.ones(len(st_mtx)), params) #stop times
    
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

#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42  
#%% firing rate vs body speed, start/stop, phase - Figure 3A,B
'''
2 example neurons
M2216 - 20
F2208 - 21
'''
stride_speed = gait_trimed['lr']['stride_velocity'].to_numpy()
# get theta used for phase domain plot
theta1 = np.linspace(0,720,params['n_circ_bins']*2) #for linear plot, 2 cycles

if mouse_id=='M2216':
    i = 20 #neuron_index
elif mouse_id=='F2208':
    i = 21
else:
    i=0 #you can change neuron index here
fr_i = binning_fr[:,i]
se_i = binning_se[:,i]
zeros_id = np.where(fr_i==0)[0]

# get 2 cycles radii
lf_radii1 = np.concatenate((svm_sp_hmtx_str['lf'][i,:], \
                  svm_sp_hmtx_str['lf'][i,:]),axis=0)
lf_radii_sm1 = gaussian_filter(lf_radii1, sigma = 2, mode='wrap')
lr_radii1 = np.concatenate((svm_sp_hmtx_str['lr'][i,:], \
                  svm_sp_hmtx_str['lr'][i,:]),axis=0)
lr_radii_sm1 = gaussian_filter(lr_radii1, sigma = 2, mode='wrap')
rf_radii1 = np.concatenate((svm_sp_hmtx_str['rf'][i,:], \
                  svm_sp_hmtx_str['rf'][i,:]),axis=0)
rf_radii_sm1 = gaussian_filter(rf_radii1, sigma = 2, mode='wrap')
rr_radii1 = np.concatenate((svm_sp_hmtx_str['rr'][i,:], \
                  svm_sp_hmtx_str['rr'][i,:]),axis=0)
rr_radii_sm1 = gaussian_filter(rr_radii1, sigma = 2, mode='wrap')

fig = plt.figure(figsize=(18,6))
# plot phase-FR
ax1 = fig.add_subplot(131)
ax1.plot(theta1,lr_radii_sm1, color = 'k', lw=1)
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_xlim([0,270])
ax1.set_xticks([0,180,360,540,720])
ax1.set_xlabel('LR limb phase (deg)', fontsize=24, family='arial')
ax1.set_ylabel('firing rate (Hz)', fontsize=24, family='arial')
ax1.tick_params(direction='out', width=1, length=5, labelsize=24)
# save phase-FR data for graphpad prism
phase_fr = zip(theta1.tolist(), \
               lf_radii_sm1.tolist(), \
               lr_radii_sm1.tolist(), \
               rf_radii_sm1.tolist(),\
               rr_radii_sm1.tolist())
phase_fr_df = pd.DataFrame(phase_fr,columns=['phase','LF','LR','RF','RR'])
#phase_fr_df.to_csv(f'{output}fr2phase_{mouse_id[-1]}_{i}.csv')

# plot speed-FR
ax2 = fig.add_subplot(132)
ax2.plot(binning_speed, fr_i, color='#000000', lw=1)
ax2.fill_between(binning_speed, fr_i+se_i,fr_i-se_i, \
                 color = '#000000', alpha=.3, linewidth=0)
ax2.set_xlim([0,300])
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_xticks([0,100,200,300])
ax2.set_xlabel('body speed (mm/s)', fontsize=24, family='arial')
ax2.set_ylabel('firing rate (Hz)', fontsize=24, family='arial')
ax2.tick_params(direction='out', width=1, length=5, labelsize=24)
# save speed-FR data for graphpad prism
sp_fr = zip(binning_speed.tolist(), fr_i.tolist(), se_i.tolist())
sp_fr_df = pd.DataFrame(sp_fr, columns=['body speed', 'firing rate', 'sem'])
#sp_fr_df.to_csv(f'{output}fr2speed_{mouse_id[-1]}_{i}.csv')

# plot start/stop-FR
ax3 = fig.add_subplot(133)
ax3.plot(t, psth_start[i,:], label='start', color='#0000FF', lw=1)
ax3.fill_between(t, psth_start[i,:]+start_se[i,:],\
                 psth_start[i,:]-start_se[i,:], \
                 color = '#0000FF', alpha=.3, linewidth=0)
ax3.plot(t, psth_stop[i,:], label='stop', color='#FF0000', lw=1)
ax3.fill_between(t, psth_stop[i,:]+stop_se[i,:],\
                 psth_stop[i,:]-stop_se[i,:], \
                 color='#FF0000', alpha=.3, linewidth=0)
ax3.set_xlim([-1,1])
if mouse_id=='F2208':
    ax3.set_ylim([0,5])
else:
    ax3.set_ylim([0,30])
ax3.set_xticks([-1,-0.5,0,0.5,1])
ax3.spines[['top', 'right']].set_visible(False)
ax3.set_xlabel('time from start/stop (s)', fontsize=24, family='arial')
ax3.set_ylabel('firing rate (Hz)', fontsize=24, family='arial')
ax3.tick_params(direction='out', width=1, length=5, labelsize=24)
ax3.legend()
# save data for graphpad prism
start_fr = zip(t.tolist(), psth_start[i,:].tolist(), start_se[i,:].tolist())
start_fr_df = pd.DataFrame(start_fr,columns=['time', 'firing rate', 'sem'])
#start_fr_df.to_csv(f'{output}fr2start_{mouse_id[-1]}_{i}.csv')
stop_fr = zip(t.tolist(), psth_stop[i,:].tolist(), stop_se[i,:].tolist())
stop_fr_df = pd.DataFrame(stop_fr,columns=['time', 'firing rate', 'sem'])
#stop_fr_df.to_csv(f'{output}fr2stop_{mouse_id[-1]}_{i}.csv')

plt.tight_layout()
if mouse_id=='F2208':
    plt.savefig(f'{figpath}Figure3-B.pdf',dpi=300,bbox_inches='tight', transparent=True)
elif mouse_id=='M2216':
    plt.savefig(f'{figpath}Figure3-A.pdf',dpi=300,bbox_inches='tight', transparent=True)
else:
    plt.savefig(f'{figpath}Figure3-{mouse_id}.pdf',dpi=300,bbox_inches='tight', transparent=True)
