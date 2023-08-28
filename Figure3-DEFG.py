# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:11:06 2023
Figure 3. Dorsal striatal neurons encoding single-limb phase angle, body speed,
 and start/stop of walking.

@author: lonya
"""
#%% Load Data
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy.stats import zscore

import pickle
#%% Load all single neuron mtx

root = tk.Tk()
filez = fd.askopenfilenames(parent=root, \
                            title='Choose all-single-neuron-mtx.pickle')
root.destroy()
print(filez)

with open(filez[0], 'rb') as f:
    stack_file = pickle.load(f)

all_binning_fr = stack_file[0]
all_psth_start = stack_file[1]
all_psth_stop = stack_file[2]
all_start_p = stack_file[3]
all_stop_p = stack_file[4]
all_rval = stack_file[5]
all_pval = stack_file[6]
all_lf = stack_file[7]
all_lr = stack_file[8]
all_rf = stack_file[9]
all_rr = stack_file[10]
all_lf_va = stack_file[11]
all_lr_va = stack_file[12]
all_rf_va = stack_file[13]
all_rr_va = stack_file[14]
all_start_speed = stack_file[15]
all_stop_speed = stack_file[16]
binning_speed = stack_file[17]
all_group = stack_file[18]
all_d1_d2 = stack_file[19]
 
#%% Load neuron_psi
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, \
                            title='Select all-mice-index.pickle')
root.destroy()
print(filez)

with open(filez[0], 'rb') as f:
    neuron_psi = pickle.load(f)

#%% create path for figures
backslash_id = [i for i,x in enumerate(filez[0]) if x=='/']

output_path = filez[0][:backslash_id[-1]+1]

figpath = str(output_path) + 'Figure 3/'

if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
#%% prepare psi data
mouse_id = neuron_psi['mouse_id'].values
mouse_id = np.unique(mouse_id)
#find neuron with vector length > 0.9, remove
vector_length = 0.9
psi_trimed = pd.DataFrame()
remove_tag = np.empty((1,))
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
    temp_tag = np.zeros(len(temp_psi_lf))
    
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
        
    temp_tag[remove_id] = 1
    remove_tag = np.append(remove_tag, temp_tag)
    
#remove_tag = np.delete(remove_tag,0,None) #remove the first element

#%%  remove bad neurons
bad_nn = np.where(remove_tag==1)[0] #bad neurons

all_psth_start_trimed = np.delete(all_psth_start, bad_nn, axis=0)
all_psth_stop_trimed = np.delete(all_psth_stop, bad_nn, axis=0)

all_binning_fr_trimed = np.delete(all_binning_fr, bad_nn, axis=1)

all_lf_trimed = np.delete(all_lf, bad_nn, axis=0)
all_lr_trimed = np.delete(all_lr, bad_nn, axis=0)
all_rf_trimed = np.delete(all_rf, bad_nn, axis=0)
all_rr_trimed = np.delete(all_rr, bad_nn, axis=0)

all_lf_va_trimed = np.delete(all_lf_va, bad_nn, axis=0)
all_lr_va_trimed = np.delete(all_lr_va, bad_nn, axis=0)
all_rf_va_trimed = np.delete(all_rf_va, bad_nn, axis=0)
all_rr_va_trimed = np.delete(all_rr_va, bad_nn, axis=0)

all_pval_trimed = np.delete(all_pval, bad_nn, axis=0)
all_start_p_trimed = np.delete(all_start_p, bad_nn, axis=0)
all_stop_p_trimed = np.delete(all_stop_p, bad_nn, axis=0)

all_group_trimed = np.delete(all_group, bad_nn, axis=0)
all_d1_d2_trimed = np.delete(all_d1_d2, bad_nn, axis=0)
#%% select healthy group
group = 'HL'

select_psi = psi_trimed.loc[psi_trimed['limb']=='lf'].sort_values(by=['index'])
select_psi = select_psi.reset_index()
select_psi = select_psi.drop(columns=['level_0','index'])

psi_lf = psi_trimed.loc[(psi_trimed['limb']=='lf')&\
                        (psi_trimed['group_id']==group)].sort_values(by=['index'])
psi_lf = psi_lf.reset_index()
psi_p_hl_lf = psi_lf['p'].values

psi_lr = psi_trimed.loc[(psi_trimed['limb']=='lr')&\
                        (psi_trimed['group_id']==group)].sort_values(by=['index'])
psi_lr = psi_lr.reset_index()
psi_p_hl_lr = psi_lr['p'].values

psi_rf = psi_trimed.loc[(psi_trimed['limb']=='rf')&\
                        (psi_trimed['group_id']==group)].sort_values(by=['index'])
psi_rf = psi_rf.reset_index()
psi_p_hl_rf = psi_rf['p'].values

psi_rr = psi_trimed.loc[(psi_trimed['limb']=='rr')&\
                        (psi_trimed['group_id']==group)].sort_values(by=['index'])
psi_rr = psi_rr.reset_index()
psi_p_hl_rr = psi_rr['p'].values

select_index = select_psi.loc[select_psi['group_id']==group].index

all_psth_start_si = all_psth_start_trimed[select_index,:]
all_psth_stop_si = all_psth_stop_trimed[select_index,:]

all_binning_fr_si = all_binning_fr_trimed[:,select_index]

all_lf_si = all_lf_trimed[select_index,:]
all_lr_si = all_lr_trimed[select_index,:]
all_rf_si = all_rf_trimed[select_index,:]
all_rr_si = all_rr_trimed[select_index,:]

all_lf_va_si = all_lf_va_trimed[select_index]
all_lr_va_si = all_lr_va_trimed[select_index]
all_rf_va_si = all_rf_va_trimed[select_index]
all_rr_va_si = all_rr_va_trimed[select_index]

all_pval_si = all_pval_trimed[select_index]
all_start_p_si = all_start_p_trimed[select_index]
all_stop_p_si = all_stop_p_trimed[select_index]

all_group_si = all_group_trimed[select_index]
all_d1_d2_si = all_d1_d2_trimed[select_index]

#%% plot stacked figure - Figure 3E

binning_fr_z = zscore(all_binning_fr_si, axis=0)
binning_fr_max_id = np.argmax(binning_fr_z,axis=0)

fr_order = np.argsort(-1*binning_fr_max_id) # ordered by speed with maximum fr
binning_fr_sorted = binning_fr_z[:,fr_order]# sort by maximum firing rate 

xgrid = np.append(binning_speed, 2*max(binning_speed)-binning_speed[-2])
ygrid = np.arange(all_binning_fr_si.shape[1]+1)

fig = plt.figure(figsize=(3,3))
ax1 = fig.add_subplot(111)

pcm = ax1.pcolormesh(xgrid, ygrid, binning_fr_sorted.T, \
                     vmin=-1, vmax=1)
ax1.set_frame_on(False) # remove all spines
ax1.set_xlabel('body speed (mm/s)')
ax1.set_ylabel('neurons')
fig.colorbar(pcm, ax=ax1)

# save data for graphpad prism
nn_sp_df = pd.DataFrame(binning_fr_sorted.T)
nn_sp_df.columns = binning_speed
#nn_sp_df.to_csv(f'{output}nn2speed_ALL.csv')
plt.savefig(f'{figpath}Figure3-E.pdf',dpi=300,bbox_inches='tight', transparent=True)
#%% stack firing rate for start - Figure 3F
all_psth_start_si_stack = all_psth_start_si[:,450:551]
psth_start_z = zscore(all_psth_start_si_stack, axis=1)
psth_max_id = np.argmax(psth_start_z,axis=1)
psth_order = np.argsort(-1*psth_max_id)
psth_start_sorted = psth_start_z[psth_order,:]

bins = np.arange(-1,1.02,0.02) #binsize 20ms
xgrid = np.append(bins, 2*max(bins)-bins[-2])
ygrid = np.arange(all_psth_start_si_stack.shape[0]+1)

fig = plt.figure(figsize=(3,3))
ax1 = fig.add_subplot(111)

pcm = ax1.pcolormesh(xgrid, ygrid, psth_start_sorted, \
                     vmin=-1, vmax=1)
ax1.set_frame_on(False) # remove all spines
ax1.set_xlim([-1,1])
ax1.set_xlabel('time from start (s)')
ax1.set_ylabel('neurons')
fig.colorbar(pcm, ax=ax1)

# save data for graphpad prism
nn_start_df = pd.DataFrame(psth_start_sorted)
nn_start_df.columns = bins
#nn_start_df.to_csv(f'{output}nn2start_ALL.csv')
plt.savefig(f'{figpath}Figure3-F.pdf',dpi=300,bbox_inches='tight', transparent=True)
#%% stack firing rate for stop - Figure 3G
all_psth_stop_si_stack = all_psth_stop_si[:,450:551]
psth_stop_z = zscore(all_psth_stop_si_stack, axis=1)
psth_max_id = np.argmax(psth_stop_z,axis=1)
psth_order = np.argsort(-1*psth_max_id)
psth_stop_sorted = psth_stop_z[psth_order,:]

bins = np.arange(-1,1.02,0.02)
xgrid = np.append(bins, 2*max(bins)-bins[-2])
ygrid = np.arange(all_psth_stop_si_stack.shape[0]+1)

fig = plt.figure(figsize=(3,3))
ax1 = fig.add_subplot(111)

pcm = ax1.pcolormesh(xgrid, ygrid, psth_stop_sorted, \
                     vmin=-2, vmax=2)
ax1.set_frame_on(False) # remove all spines
ax1.set_xlim([-1,1])
ax1.set_xlabel('time from stop (s)')
ax1.set_ylabel('neurons')
fig.colorbar(pcm, ax=ax1)

# save data for graphpad prism
nn_stop_df = pd.DataFrame(psth_stop_sorted)
nn_stop_df.columns = bins
#nn_stop_df.to_csv(f'{output}nn2stop_ALL.csv')
plt.savefig(f'{figpath}Figure3-G.pdf',dpi=300,bbox_inches='tight', transparent=True)
#%% stack firing rate for limb phase - Figure 3D
theta1 = np.linspace(0,720,48,endpoint=False) #for linear plot, 2 cycles

lf_radii1 = np.append(all_lf_si, all_lf_si,axis=1)
lf_radii_sm1 = gaussian_filter1d(lf_radii1, sigma = 2, axis = 1, mode='wrap')
lr_radii1 = np.append(all_lr_si, all_lr_si,axis=1)
lr_radii_sm1 = gaussian_filter1d(lr_radii1, sigma = 2, axis = 1, mode='wrap')
rf_radii1 = np.append(all_rf_si, all_rf_si,axis=1)
rf_radii_sm1 = gaussian_filter1d(rf_radii1, sigma = 2, axis = 1, mode='wrap')
rr_radii1 = np.append(all_rr_si, all_rr_si,axis=1)
rr_radii_sm1 = gaussian_filter1d(rr_radii1, sigma = 2, axis = 1, mode='wrap')

lf_radii_z = zscore(lf_radii_sm1, axis=1)
lf_order = np.argsort(all_lf_va_si)
lf_radii_sorted = lf_radii_z[lf_order,:]

lr_radii_z = zscore(lr_radii_sm1, axis=1)
lr_order = np.argsort(all_lr_va_si)
lr_radii_sorted = lr_radii_z[lr_order,:]

rf_radii_z = zscore(rf_radii_sm1, axis=1)
rf_order = np.argsort(all_rf_va_si)
rf_radii_sorted = rf_radii_z[rf_order,:]

rr_radii_z = zscore(rr_radii_sm1, axis=1)
rr_order = np.argsort(all_rr_va_si)
rr_radii_sorted = rr_radii_z[rr_order,:]

xgrid = np.append(theta1, 2*max(theta1)-theta1[-2])
ygrid = np.arange(lf_radii_z.shape[0]+1)

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(221)
pcm = ax1.pcolormesh(xgrid, ygrid, lf_radii_sorted)
ax1.set_frame_on(False) # remove all spines
ax1.set_xlabel('limb phase (deg)')
ax1.set_ylabel('neurons')
ax1.set_title('LF')
fig.colorbar(pcm, ax=ax1)

# save data for graphpad prism
lf_phase_df = pd.DataFrame(lf_radii_sorted)
lf_phase_df.columns = theta1
#lf_phase_df.to_csv(f'{output}nn2phase_LF.csv')

ax2 = fig.add_subplot(222)
pcm = ax2.pcolormesh(xgrid, ygrid, rf_radii_sorted)
ax2.set_frame_on(False) # remove all spines
ax2.set_xlabel('limb phase (deg)')
ax2.set_ylabel('neurons')
ax2.set_title('RF')
fig.colorbar(pcm, ax=ax2)

# save data for graphpad prism
lr_phase_df = pd.DataFrame(lr_radii_sorted)
lr_phase_df.columns = theta1
#lr_phase_df.to_csv(f'{output}nn2phase_LR.csv')

ax3 = fig.add_subplot(223)
pcm = ax3.pcolormesh(xgrid, ygrid, lr_radii_sorted)
ax3.set_frame_on(False) # remove all spines
ax3.set_xlabel('limb phase (deg)')
ax3.set_ylabel('neurons')
ax3.set_title('LR')
fig.colorbar(pcm, ax=ax3)

# save data for graphpad prism
rf_phase_df = pd.DataFrame(rf_radii_sorted)
rf_phase_df.columns = theta1
#rf_phase_df.to_csv(f'{output}nn2phase_RF.csv')

ax4 = fig.add_subplot(224)
pcm = ax4.pcolormesh(xgrid, ygrid, rr_radii_sorted)
ax4.set_frame_on(False) # remove all spines
ax4.set_xlabel('limb phase (deg)')
ax4.set_ylabel('neurons')
ax4.set_title('RR')
fig.colorbar(pcm, ax=ax4)

# save data for graphpad prism
rr_phase_df = pd.DataFrame(rr_radii_sorted)
rr_phase_df.columns = theta1
#rr_phase_df.to_csv(f'{output}nn2phase_RR.csv')

plt.tight_layout()
plt.savefig(f'{figpath}Figure3-D.pdf',dpi=300,bbox_inches='tight', transparent=True)