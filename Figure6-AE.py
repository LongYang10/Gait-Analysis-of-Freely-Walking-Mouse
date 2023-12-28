# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:11:11 2023
@author: lonya
"""
#%% Load Data
from tkinter import messagebox
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
import scipy.io as sio
import my_funcs as mf
#%%
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Choose F1590.mat or F1920.mat or M1579.mat or F2275.mat',\
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
#%% create path for figures
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']
output_path = all_files[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 6/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% processing data
print('load file: ' + all_files[0])
mouse_id = all_files[0][len(all_files[0])-9:-4]

neuron_psi = pd.DataFrame()
neuron_index = pd.DataFrame()

if mouse_id=='F1590':
    strain='D1'
    group_id='CT'
    stage='tagging'
elif mouse_id=='F1920':
    strain='D2'
    group_id='CT'
    stage='tagging'
elif mouse_id=='F1579':
    strain='D1'
    group_id='PD'
    stage='tagging'
elif mouse_id=='F2275':
    strain='D2'
    group_id='PD'
    stage='tagging'
else:
    strain=''
    group_id=''
    stage=''

matfile = sio.loadmat(all_files[0], squeeze_me = True)
eMouse = matfile['eMouse']

# get stimuli
pulseTimes = eMouse['stimuli'].item()['pulseTimes'].item()
laserOn = eMouse['stimuli'].item()['pulseTrainTimes'].item()

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
    
# get opto-tagging data
new_tagging, tagss = mf.get_tagging(st_mtx, clwf, fs, pulseTimes, params)

# calculate phase locking
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
    
#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% plot raster, pre/post laser waveform of tagged neuron - Figure 6AE
'''
2 example neurons:
sham lesion: F1590; D1 neuron: 17
sham lesion: F1920; D2 neuron: 4

6OHDA lesion: M1579; D1 neuron: 34
6OHDA lesion: F2275; D2 neuron: 11
'''

theta1 = np.linspace(0,720,params['n_circ_bins']*2) #for linear plot, 2 cycles
# example neuron selected
if mouse_id=='F1590':
    i=17
    k=17
elif mouse_id=='F1920':
    i=4
    k=4
elif mouse_id=='M1579':
    i=34
    k=34
elif mouse_id=='F2275':
    i=11
    k=11
else:
    print('default neuron index=0')

# get 2 cycles radii & the smoothed firing rate in phase domain
lr_radii1 = np.concatenate((svm_sp_hmtx_str['lr'][k,:], \
                      svm_sp_hmtx_str['lr'][k,:]),axis=0)
lr_radii_sm1 = gaussian_filter(lr_radii1, sigma = 2, mode='wrap')
    
fig = plt.figure(figsize=(6,6))
ax3 = fig.add_subplot(111)
ax3.plot(theta1,lr_radii_sm1/np.min(lr_radii_sm1), color = '#00FF00', lw=1)
ax3.spines[['top', 'right']].set_visible(False)
ax3.set_xlim([0,270])
ax3.set_xticks([0,180,360,540,720])
ax3.set_xlabel('limb phase (deg)', fontsize=18, family='arial')
ax3.set_ylabel('normalized firing rate (Hz)', fontsize=18, family='arial')
ax3.tick_params(direction='out', width=1, length=5, labelsize=12)
# save data for graphpad prism
fr2ph = zip(theta1.tolist(),lr_radii_sm1.tolist())
fr2ph_df = pd.DataFrame(fr2ph,columns=['phase','lr'])
#fr2ph_df.to_csv(f'{output}fr2phase{mouse_id[-1]}_{i}.csv')
lr_radii_sm1_zs = lr_radii_sm1/np.min(lr_radii_sm1)
fr2ph = zip(theta1.tolist(),lr_radii_sm1_zs.tolist())
fr2ph_df = pd.DataFrame(fr2ph,columns=['phase','lr'])
#fr2ph_df.to_csv(f'{output}fr2phase{mouse_id[-1]}_{i}_zscore.csv')

if mouse_id=='F1590':
    plt.savefig(f'{figpath}Figure6-A-D1.pdf', dpi=300,bbox_inches='tight', transparent=True)
elif mouse_id=='F1920':
    plt.savefig(f'{figpath}Figure6-A-D2.pdf', dpi=300,bbox_inches='tight', transparent=True)
elif mouse_id=='M1579':
    plt.savefig(f'{figpath}Figure6-D-D1.pdf', dpi=300,bbox_inches='tight', transparent=True)
elif mouse_id=='F2275':
    plt.savefig(f'{figpath}Figure6-D-D2.pdf', dpi=300,bbox_inches='tight', transparent=True)
else:
    plt.savefig(f'{figpath}Figure6-{mouse_id}.pdf', dpi=300,bbox_inches='tight', transparent=True)