# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:59:08 2023

@author: lonya
"""

#%% Load Data
import tkinter as tk
import tkinter.filedialog as fd
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import my_funcs as mf
import pickle
import seaborn as sns

#%% Load neuron_psi
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, \
                            title='Select all-mice-index.pickle')
root.destroy()
print(filez)

with open(filez[0], 'rb') as f:
    neuron_psi = pickle.load(f)
# turn phase angle from radian to degree
neuron_psi['a'] = neuron_psi['a']*180/np.pi

#%% create path for figures
backslash_id = [i for i,x in enumerate(filez[0]) if x=='/']

output_path = filez[0][:backslash_id[-1]+1]

figpath = str(output_path) + 'Figure 2/'

if not os.path.exists(figpath):
    os.makedirs(figpath)

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
   
#%%  - bar plot vector length across limbs - Figure 2I
group = 'HL'
hl_psi = psi_trimed.loc[(psi_trimed['group_id']==group)]

f, ax1 = plt.subplots(1, 1, figsize=(3, 3))

sns.barplot(data=hl_psi.dropna(), x="limb", y="r", \
            palette = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF'], \
            capsize=0.1, ci=85, errwidth=2, ax=ax1)
ax1.set_ylabel('vector length', fontsize=16, family='arial')
ax1.set_xlabel('limb',fontsize=16,family='arial')
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=16)
ax1.set_xticklabels(["LF", "LR", "RF", "RR"])
ax1.set_yticks([0, 0.05, 0.1, 0.15])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

lf_r = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lf'),'r'].values
lr_r = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lr'),'r'].values
rf_r = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rf'),'r'].values
rr_r = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rr'),'r'].values
    
# save data for graphpad prism
vl = zip(lf_r.tolist(), lr_r.tolist(), rf_r.tolist(), rr_r.tolist())
vl_df = pd.DataFrame(vl, columns=['lf', 'lr', 'rf', 'rr'])
#vl_df.to_csv(f'{output}mean_vector_length_{group}.csv')

plt.savefig(f'{figpath}Fig2-I.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% bar plot of mean vector angles of all neurons - healthy group - Figure 2J
group = 'HL'
lf_a = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lf'),['a']].values
lf_m, lf_sd, _ = mf.circ_m(lf_a)

lr_a = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lr'),['a']].values
lr_m, lr_sd, _ = mf.circ_m(lr_a)

rf_a = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rf'),['a']].values
rf_m, rf_sd, _ = mf.circ_m(rf_a)

rr_a = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rr'),['a']].values
rr_m, rr_sd, _ = mf.circ_m(rr_a)

# angular permutation test
a_p_lf_lr = mf.permut_angle(lf_a, lr_a, 1000)
a_p_lr_rr = mf.permut_angle(lr_a, rr_a, 1000)
a_p_lf_rf = mf.permut_angle(lf_a, rf_a, 1000)
a_p_lf_rr = mf.permut_angle(lf_a, rr_a, 1000)
a_p_lr_rf = mf.permut_angle(lr_a, rf_a, 1000)
a_p_rf_rr = mf.permut_angle(rf_a, rr_a, 1000)

mean_a = (lf_m, lr_m, rf_m, rr_m)
std_a = np.array([lf_sd, lr_sd, rf_sd, rr_sd])
ind = np.arange(4)  
width = 0.8 

f, ax2 = plt.subplots(1, 1, figsize=(3, 3))

ax2.bar(ind, mean_a, width, yerr = std_a, capsize = 6,\
        error_kw=dict(capthick=2), \
        color = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF'])
ax2.set_ylabel('vector angle (deg)', fontsize=16, family='arial')
ax2.set_xlabel('limb',fontsize=16,family='arial')
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=16)
ax2.set_xticks(ind,("LF", "LR", "RF", "RR"))
#ax2.set_yticks([0, 0.5, 1, 1.5, 2])
ax2.set_yticks([0, 90, 180, 270, 360])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')
#plt.savefig(f'{output}vector_angle_healthy.png', dpi=300, \
#            bbox_inches='tight', transparent=True)
#plt.tight_layout()

# save data for graphpad prism
va = zip(list(mean_a), list(std_a))
va_df = pd.DataFrame(va, columns=['mean','std'])
#va_df.to_csv(f'{output}mean_vector_angle_{group}.csv')
plt.savefig(f'{figpath}Fig2-J.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% get phase locked neurons - Figure 2H
#Figure 2H
group = 'HL'

select_psi = psi_trimed.loc[psi_trimed['group_id']==group].sort_values(by=['index'])
select_psi = select_psi.reset_index()
select_psi = select_psi.drop(columns=['level_0','index'])

#select_psi = select_psi.loc[(neuron_psi['group_id']=='HL')]
lf_perc = 100*len(select_psi[(select_psi['limb']=='lf') & \
                         (select_psi['p']<=0.05)])/len(select_psi[select_psi['limb']=='lf'])
lr_perc = 100*len(select_psi[(select_psi['limb']=='lr') & \
                         (select_psi['p']<=0.05)])/len(select_psi[select_psi['limb']=='lr'])
rf_perc = 100*len(select_psi[(select_psi['limb']=='rf') & \
                         (select_psi['p']<=0.05)])/len(select_psi[select_psi['limb']=='rf'])
rr_perc = 100*len(select_psi[(select_psi['limb']=='rr') & \
                         (select_psi['p']<=0.05)])/len(select_psi[select_psi['limb']=='rr'])
p1 = select_psi.loc[select_psi['limb']=='lf',['p']].values
p2 = select_psi.loc[select_psi['limb']=='lr',['p']].values
p3 = select_psi.loc[select_psi['limb']=='rf',['p']].values
p4 = select_psi.loc[select_psi['limb']=='rr',['p']].values
p_sum= p1+p2+p3+p4
all_perc = 100*len(p_sum[(p1<=0.05) | (p2<=0.05) | \
                         (p3<=0.05) | (p4<=0.05)])/len(select_psi[select_psi['limb']=='lf'])

import matplotlib.pyplot as plt
# creating the dataset
data = {'LF':lf_perc, 'LR':lr_perc, 'RF':rf_perc, 'RR':rr_perc, 'ALL': all_perc}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize=(3, 3))
ax1 = fig.add_subplot(111)
 
# creating the bar plot
ax1.bar(courses, values, color =['#FF0000', '#00FF00', '#0000FF', '#FF00FF','k'],
        width = 0.7)

ax1.set_yticks([0, 10, 20, 30, 40,50])
ax1.set_ylabel("% of phase locked cells", fontsize=16, family='arial')
ax1.set_xlabel('limb',fontsize=16,family='arial')
ax1.spines[['left','bottom']].set_linewidth(1)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=1, length=5, labelsize=16)
ax1.set_xticklabels(["LF", "LR", "RF", "RR", "≥1"])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

# save data for graphpad prism
pn = zip(courses, values)
pn_df = pd.DataFrame(pn, columns=['limb','N'])
#pn_df.to_csv(f'{output}phase-locked-neurons_{group}.csv')

plt.savefig(f'{figpath}Fig2-H.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% plot all neurons' mean vector angle distribution - Figure 2K
group = 'HL'
nbin = 10

lf_phase = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lf'),['r','a']].values
for i in range(lf_phase.shape[0]):
    if lf_phase[i,1]<0:
        lf_phase[i,1] = lf_phase[i,1]+360
lr_phase = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lr'),['r','a']].values
for i in range(lr_phase.shape[0]):
    if lr_phase[i,1]<0:
        lr_phase[i,1] = lr_phase[i,1]+360
rf_phase = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rf'),['r','a']].values
for i in range(rf_phase.shape[0]):
    if rf_phase[i,1]<0:
        rf_phase[i,1] = rf_phase[i,1]+360
rr_phase = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rr'),['r','a']].values
for i in range(rr_phase.shape[0]):
    if rr_phase[i,1]<0:
        rr_phase[i,1] = rr_phase[i,1]+360

lf_hist,bin_edges = np.histogram(np.concatenate((lf_phase[:,1],lf_phase[:,1]+360)),\
         nbin,range=[0,720])
bins_x = bin_edges[:-1]
lf_hist = 2*lf_hist/sum(lf_hist)*100
    
lr_hist,_ = np.histogram(np.concatenate((lr_phase[:,1],lr_phase[:,1]+360)),\
         nbin,range=[0,720])
lr_hist = 2*lr_hist/sum(lr_hist)*100

rf_hist,_ = np.histogram(np.concatenate((rf_phase[:,1],rf_phase[:,1]+360)),\
         nbin,range=[0,720])
rf_hist = 2*rf_hist/sum(rf_hist)*100

rr_hist,_ = np.histogram(np.concatenate((rr_phase[:,1],rr_phase[:,1]+360)),\
         nbin,range=[0,720])
rr_hist = 2*rr_hist/sum(rr_hist)*100
# save data for graphpad prism
phase_hist_fr = zip(bins_x.tolist(), lf_hist.tolist(), lr_hist.tolist(),\
                    rf_hist.tolist(), rr_hist.tolist())
phase_hist_fr_df = pd.DataFrame(phase_hist_fr, \
                                columns=['angle', 'LF', 'LR', 'RF','RR'])
#phase_hist_fr_df.to_csv(f'{output}phase_distribution_all_neuron_4limbs.csv')

# plot the histogram
colors = ['#FF0000','#00FF00','#0000FF','#FF00FF']
fig, ax = plt.subplots(1,1,figsize=(6,6))
phase_hist_fr_df.plot(ax=ax,x='angle',color=colors)
ax.set_box_aspect(1)
ax.set_xlim([0,720])
ax.set_xticks([0,180,360,540,720])
ax.set_ylim([10,35])
ax.set_yticks([10,20,30])
ax.tick_params(direction='out', width=1, length=10, labelsize=24)
ax.set_xlabel('preferred limb phase (deg)', fontsize=24, family='arial')
ax.set_ylabel('% of cells', fontsize=24, family='arial')
ax.spines[['top', 'right']].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
plt.savefig(f'{figpath}Fig2-K.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% get phase locked neurons in 1 limb, 2 limb, 3 limbs, 4 limbs
#new Figure 2 – Figure Supplement 1A
group = 'HL'

p1 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                    (psi_trimed['limb']=='lf'),['p']].values
p2 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                    (psi_trimed['limb']=='lr'),['p']].values
p3 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                    (psi_trimed['limb']=='rf'),['p']].values
p4 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                    (psi_trimed['limb']=='rr'),['p']].values
limb_p = np.concatenate((p1, p2, p3, p4),axis=1)
limb_p_tag = np.zeros_like(limb_p)
limb_p_tag[np.where(limb_p<0.05)]=1
limb_p_sum = np.sum(limb_p_tag, axis=1)

# find diagnal limb
double_limb_i = np.where(limb_p_sum==2)[0]
diag_limb = 0
non_diag = 0
for i in range(len(double_limb_i)):
    row_i = double_limb_i[i]
    if ((limb_p_tag[row_i, 0]==1) & (limb_p_tag[row_i,3]==1)) | \
       ((limb_p_tag[row_i, 1]==1) & (limb_p_tag[row_i,2]==1)):
        diag_limb = diag_limb+1
    else:
        non_diag = non_diag+1

import matplotlib.pyplot as plt

labels = 'none', '1 limb', '2 limbs', '3 limbs', '4 limbs'
sizes = [len(np.where(limb_p_sum==0)[0]), len(np.where(limb_p_sum==1)[0]), \
         len(np.where(limb_p_sum==2)[0]), len(np.where(limb_p_sum==3)[0]), \
         len(np.where(limb_p_sum==4)[0])]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')


# creating the dataset
sizes_pt = np.asarray(sizes)/sum(sizes)*100
data = {'1':sizes_pt[1], '2':sizes_pt[2], '3':sizes_pt[3], '4': sizes_pt[4]}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize=(3, 3))
ax1 = fig.add_subplot(111)
 
# creating the bar plot
ax1.bar(courses, values, color =['#FF0000', '#00FF00', '#0000FF', '#FF00FF'],
        width = 0.7)

ax1.set_yticks([0,5, 10,15, 20])
ax1.set_ylabel("% of cells", fontsize=16, family='arial')
ax1.set_xlabel('number of limb',fontsize=16,family='arial')
ax1.spines[['left','bottom']].set_linewidth(1)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=1, length=5, labelsize=16)
ax1.set_xticklabels(["1", "2", "3", "4"])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

# save data for graphpad prism
pn_nl = zip(courses, values)
pn_nl_df = pd.DataFrame(pn_nl, columns=['limb','N'])
#plt.savefig(f'{figpath}Fig2-A-{group}-si.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% correlation between firing rate and vector length - single limb
# new Figure 2 – Figure Supplement 1E
from scipy.optimize import curve_fit
group = 'HL'
limb = 'lf'
# phase locking encoding (at least one limb)
vl_lf = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']==limb),['r']].values.squeeze() 
nfr = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                     (psi_trimed['limb']==limb),['base_fr']].values.squeeze() 
    
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
#ax.scatter(vl_lf, nfr)
def linear_func(x, a, b):
    return b + a * x 
# calculate the linear fitting of speed data
popt, pcov = curve_fit(linear_func, np.log(nfr), np.log(vl_lf))

x_data = np.linspace(min(nfr), max(nfr), 300)
y_fit = linear_func(np.log(x_data), *popt)

# plot firing rate vs vector length
ax.plot(nfr,vl_lf, '.k', ms=6)
ax.plot(x_data, np.exp(y_fit), ':r', lw=3)
#ax.set_xlim([0,400])
#ax.set_ylim([0,400])
ax.tick_params(direction='out', width=1, length=10, labelsize=24)
ax.set_xlabel('firing rate (1/s)', fontsize=24, family='arial')
ax.set_ylabel('vector length (n.s.)', fontsize=24, family='arial')
ax.spines[['top', 'right']].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
plt.xscale("log")
plt.yscale('log')
#plt.savefig(f'{figpath}Fig2-si-E.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% vector length rank on limb
# new Figure 2 – Figure Supplement 1C

group = 'HL'
# phase locking encoding (at least one limb)
vl_lf = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']=='lf'),['r']].values.squeeze()
vl_lr = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']=='lr'),['r']].values.squeeze()
vl_rf = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']=='rf'),['r']].values.squeeze() 
vl_rr = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']=='rr'),['r']].values.squeeze() 
    
vl_rank = np.concatenate((np.expand_dims(vl_lf, axis=1), \
                          np.expand_dims(vl_lr, axis=1), \
                          np.expand_dims(vl_rf, axis=1), \
                          np.expand_dims(vl_rr, axis=1)), axis=1)
vl_sort = np.sort(vl_rank, axis=1)

# creating the dataset
data = {'rank': ["1st" for x in range(len(vl_lf))]+\
                ["2nd" for x in range(len(vl_lf))]+\
                ["3rd" for x in range(len(vl_lf))]+\
                ["4th" for x in range(len(vl_lf))],
        'vector length':vl_sort[:,3].tolist()+vl_sort[:,2].tolist()+\
        vl_sort[:,1].tolist()+vl_sort[:,0].tolist()}
  
fig = plt.figure(figsize=(3, 3))
ax1 = fig.add_subplot(111)
sns.barplot(data, x="rank", y="vector length", errorbar="se",ax=ax1)
ax1.set_ylabel('vector length', fontsize=12, family='arial')
ax1.set(xlabel='ranked limb')
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=12)
ax1.set_xticklabels(["1st", "2nd", "3rd", "4th"])
ax1.set_yticks([0.0, 0.05, 0.1, 0.15])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')
plt.tight_layout()

#plt.savefig(f'{figpath}Fig2-si-C.pdf', dpi=300, bbox_inches='tight', transparent=True)
