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

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
#%%
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                        title='Choose F2203.mat or F2178.mat',\
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

figpath = str(output_path) + 'Figure 4/'

if not os.path.exists(figpath):
    os.makedirs(figpath)
    
#%% processing data
print('load file: ' + all_files[0])
mouse_id = all_files[0][len(all_files[0])-9:-4]

if mouse_id=='F2203':
    strain='D1'
    group_id='HL'
    stage='tagging'
elif mouse_id=='F2178':
    strain='D2'
    group_id='HL'
    stage='tagging'
else:
    strain=''
    group_id=''
    stage=''
    
neuron_psi = pd.DataFrame()
neuron_index = pd.DataFrame()
    
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
    
#psi = mf.get_stride_psi(svm_angle, st_mtx, stride_trimed)
params['repeat'] = 1 #no jitting test, save time
stride_index = mf.get_stride_index(svm_angle, kinematics,\
                                   st_mtx, stride_trimed, params)
# create data frame
#psi_df = mf.get_psi_df(psi, tagging, xcoords, ycoords, unitBestCh,\
#                       mouse_id[-1], group_id[-1], strain[-1], \
#                       stage[-1], kw = tagss[['latency','baseline_fr']])
index_df = mf.get_stride_df(stride_index, tagging, xcoords, ycoords, unitBestCh,\
                       mouse_id, group_id[-1], strain[-1], \
                       stage[-1], kw = tagss[['latency','baseline_fr']])
#neuron_psi = pd.concat([neuron_psi, psi_df], ignore_index = True)
neuron_index = pd.concat([neuron_index, index_df], ignore_index = True)

#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% 
neuron_psi = neuron_index

#%% prepare psi data
mouse_id = neuron_psi['mouse_id'].values
mouse_id = np.unique(mouse_id)
#find neuron with vector length > 0.9, remove
vector_length = 0.9
psi_trimed = pd.DataFrame()
for i in range(len(mouse_id)):
    mouse = mouse_id[i]
    temp_psi_lf = neuron_psi.loc[(neuron_psi['mouse_id']==mouse)&\
                                     (neuron_psi['limb']=='lf')].reset_index()
    temp_psi_lr = neuron_psi.loc[(neuron_psi['mouse_id']==mouse)&\
                                     (neuron_psi['limb']=='lr')].reset_index()    
    temp_psi_rf = neuron_psi.loc[(neuron_psi['mouse_id']==mouse)&\
                                     (neuron_psi['limb']=='rf')].reset_index()
    temp_psi_rr = neuron_psi.loc[(neuron_psi['mouse_id']==mouse)&\
                                     (neuron_psi['limb']=='rr')].reset_index()
    temp_lf_r = temp_psi_lf['r'].values
    temp_lr_r = temp_psi_lr['r'].values
    temp_rf_r = temp_psi_rf['r'].values
    temp_rr_r = temp_psi_rr['r'].values
    error_tag = np.zeros((4,len(temp_lf_r)))
    error_tag[0,np.where((temp_lf_r>=vector_length) | (np.isnan(temp_lf_r)))[0]] = 1
    error_tag[1,np.where((temp_lr_r>=vector_length) | (np.isnan(temp_lr_r)))[0]] = 1
    error_tag[2,np.where((temp_rf_r>=vector_length) | (np.isnan(temp_rf_r)))[0]] = 1
    error_tag[3,np.where((temp_rr_r>=vector_length) | (np.isnan(temp_rr_r)))[0]] = 1
    error_tag = np.sum(error_tag,axis=0)
    remove_id = np.where(error_tag>=1)[0]
    temp_psi_lf.drop(remove_id,inplace=True)
    temp_psi_lr.drop(remove_id,inplace=True)
    temp_psi_rf.drop(remove_id,inplace=True)
    temp_psi_rr.drop(remove_id,inplace=True)
    psi_trimed = pd.concat([psi_trimed, temp_psi_lf, temp_psi_lr, temp_psi_rf,\
                            temp_psi_rr], ignore_index = True)
#%% plot raster, pre/post laser waveform of tagged neuron - Figure 4AB
'''
2 example neurons:
F2203 - 19
F2178 - 21
'''
# get data relative to left rear limb
psi_lr = psi_trimed.loc[(psi_trimed['limb']=='lr') &\
                        (psi_trimed['mouse_id']==mouse_id[0])].reset_index()
window = [-0.002, 0.006] # raster plot window in second
theta1 = np.linspace(0,720,params['n_circ_bins']*2) #for linear plot, 2 cycles
wave_t0 = 34; #default=34. used to select a narrower spike waveform time window.
wave_tf = 66; #default=66.

# example neuron selected
if mouse_id=='F2203':
    i=19
    k=19
elif mouse_id=='F2178':
    i=21
    k=0
else:
    i=0
    k=0
    print('default neuron index=0')
    
st = st_mtx[i]
wf = clwf[i]
n_wf = wf[wave_t0:wave_tf,:]# narrow waveform
t_wf = np.arange(n_wf.shape[0])/fs * 1000

# get the averaged waveform
post_laser_spid = np.empty((1,))
laser_fr = np.zeros((len(pulseTimes),2))
latency1st = []
for j in range(len(pulseTimes)):
    pt = pulseTimes[j]
    pre_index = np.where((st<pt-params['min_latency'])&\
                         (st>pt-params['pre_laser']))[0]
    post_index = np.where((st>pt+params['min_latency']) &\
                               (st<pt+params['max_latency']))[0]
    laser_fr[j,0] = len(pre_index)/(params['pre_laser']-params['min_latency'])
    laser_fr[j,1] = len(post_index)/(params['max_latency']-params['min_latency'])
    if len(post_index)>0:
        post_laser_spid = np.append(post_laser_spid,post_index)
        latency1st.append(st[post_index[0]]-pt)
        
post_laser_spid = np.delete(post_laser_spid,0,None)#remove the 1st element
if len(post_laser_spid)>0:
    post_laser_spid = post_laser_spid.astype(int)
    post_wf_mean = np.mean(n_wf[:,post_laser_spid],axis=1)
else:
    post_wf_mean = np.zeros(n_wf.shape[0])
n_post = len(post_laser_spid)
base_spid = np.where(st<pulseTimes[0])[0]
base_wf_mean = np.mean(n_wf[:,len(base_spid)-n_post+1:-1], axis=1)

# get spikes
event1_spike_times = \
    mf.restrict_spike_times(st = st, event = pulseTimes, win=window)

# get 2 cycles radii & the smoothed firing rate in phase domain
lr_radii1 = np.concatenate((svm_sp_hmtx_str['lr'][k,:], \
                      svm_sp_hmtx_str['lr'][k,:]),axis=0)
lr_radii_sm1 = gaussian_filter(lr_radii1, sigma = 2, mode='wrap')
    
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(221)
ax1.eventplot(event1_spike_times,color='k',linelengths=3,linewidths=1)
ax1.set_xlim(window)
ax1.set_xticks([-0.002,0,0.002,0.004,0.006])
ax1.set_xticklabels(['-2','0','2','4','6'])
ax1.set_yticks([0,100,200]);
ax1.tick_params(direction='out', width=1, length=5, labelsize=18)
ax1.set_xlabel('time from laser on (ms)', fontsize=18, family='arial')
ax1.set_ylabel("pulse number", fontsize=18, family='arial')
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines['left'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)

ax2 = fig.add_subplot(222)
ax2.plot(t_wf, base_wf_mean, lw=2, label='pre', color='k')
ax2.plot(t_wf, post_wf_mean, lw=2, label='post', color='b')
ax2.set_ylabel('amplitude (uV)', fontsize=18, family='arial')
ax2.tick_params(direction='out', width=1, length=5, labelsize=16)
ax2.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax2.spines[['top', 'right','bottom']].set_visible(False)
fontprops = fm.FontProperties(family='arial',size=16)
scalebar = AnchoredSizeBar(ax2.transData,
                           0.5, '0.5 ms', 
                           loc = 'lower right', 
                           bbox_to_anchor=(0.5, -0.2),
                           bbox_transform=ax2.transAxes,
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1,
                           fontproperties=fontprops)

ax2.add_artist(scalebar)
# save data for graphpad prism
wavef = zip(t_wf.tolist(),base_wf_mean.tolist(),post_wf_mean.tolist())
wavef_df = pd.DataFrame(wavef,columns=['time','base','post'])
#wavef_df.to_csv(f'{output}wf2time{mouse_id[-1]}_{i}.csv')

ax3 = fig.add_subplot(223)
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

plt.tight_layout()

if mouse_id=='F2203':
    plt.savefig(f'{figpath}Figure4-A.pdf',dpi=300,bbox_inches='tight', transparent=True)
elif mouse_id=='F2178':
    plt.savefig(f'{figpath}Figure4-B.pdf',dpi=300,bbox_inches='tight', transparent=True)
else:
    plt.savefig(f'{figpath}Figure4-{mouse_id}.pdf',dpi=300,bbox_inches='tight', transparent=True)