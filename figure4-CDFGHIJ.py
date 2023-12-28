# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:11:11 2023
@author: lonya
"""
#%% Load Data
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import my_funcs as mf
import seaborn as sns
from scipy.stats import sem
from scipy.stats import zscore
import pickle
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
figpath = str(output_path) + 'Figure 4/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% Load single neuron mtx
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
    
remove_tag = np.delete(remove_tag,0,None) #remove the first element
        
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
all_rval_trimed = np.delete(all_rval, bad_nn, axis=0)
all_start_p_trimed = np.delete(all_start_p, bad_nn, axis=0)
all_stop_p_trimed = np.delete(all_stop_p, bad_nn, axis=0)

all_group_trimed = np.delete(all_group, bad_nn, axis=0)
all_d1_d2_trimed = np.delete(all_d1_d2, bad_nn, axis=0)


#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% Cumulative distribution of the latency to spiking - Figure 4C
group = 'HL'
hl_psi = psi_trimed.loc[(psi_trimed['group_id']==group)]
hl_psi_tagging = hl_psi.loc[(hl_psi['tagging']==1)]
hl_latency_d1 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D1') &\
                            (hl_psi_tagging['limb']=='lf'),['latency']].values
hl_latency_d2 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D2') &\
                            (hl_psi_tagging['limb']=='lf'),['latency']].values
    
fig1 = plt.figure(figsize=(3,3))
ax1 = fig1.add_subplot(111)
# plot the cumulative histogram
n1, bins1, patches = ax1.hist(hl_latency_d1*1000, 15, [0,6], density=False, \
                            histtype='step', cumulative=True, label='D1',color='b')
n2,_,_ =ax1.hist(hl_latency_d2*1000, 15, [0,6], density=False, \
                            histtype='step', cumulative=True, label='D2',color='r')
ax1.set_xlabel('latency (ms)')
ax1.set_ylabel('cumulative')
ax1.legend()

# save data for graphpad prism
latency = zip(bins1[1:].tolist(),(n1/hl_latency_d1.shape[0]*100).tolist(),\
              (n2/hl_latency_d2.shape[0]*100).tolist())
latency_df = pd.DataFrame(latency,columns=['time','D1','D2'])


fig, ax = plt.subplots(1,1,figsize=(6,6))
latency_df.plot(ax=ax,x='time',color=['b','r'])
ax.set_box_aspect(1)
ax.set_xlim([0,6])
ax.set_ylim([0,100])
ax.set_yticks([0,50,100])
ax.tick_params(direction='out', width=1, length=10, labelsize=24)
ax.set_xlabel('mean spiking latency (ms)', fontsize=24, family='arial')
ax.set_ylabel('cumulative % of cells', fontsize=24, family='arial')
ax.spines[['top', 'right']].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

plt.savefig(f'{figpath}Figure4-C.pdf',dpi=300,bbox_inches='tight', transparent=True)

#%% plot the firing rate of D1 and D2 in healthy group
# new Figure4D - mean session firing rate
group = 'HL'
limb = 'lf'

d1_fr = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']==limb)&\
                       (psi_trimed['strain']=='D1')&\
                       (psi_trimed['tagging']==1),['base_fr']].values.squeeze()
d2_fr = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                       (psi_trimed['limb']==limb)&\
                       (psi_trimed['strain']=='D2')&\
                       (psi_trimed['tagging']==1),['base_fr']].values.squeeze() 
    
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(3,6))
ax1 = fig.add_subplot(111)
session_fr_fig = {
                'stain': ["D1" for x in range(len(d1_fr))]+\
                         ["D2" for x in range(len(d2_fr))],
                'fr': d1_fr.tolist() + d2_fr.tolist()
                }
sns.pointplot(data=session_fr_fig, x="stain", y="fr", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
ax1.set_ylabel('session firing rate (1/s)', fontsize=24, family='arial')
ax1.set(xlabel=None)
ax1.set_ylim([0,8])
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=24)
ax1.set_xticklabels(["D1", "D2"])
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')
plt.tight_layout()

plt.savefig(f'{figpath}Figure4D.pdf',dpi=300,bbox_inches='tight', transparent=True)

#%% D1 vs D2 vector length cross limbs - Figure 4F
group = 'HL'
hl_psi = psi_trimed.loc[(psi_trimed['group_id']==group)]
hl_psi_tagging = hl_psi.loc[(hl_psi['tagging']==1)]
n1 = len(hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D1') &\
                            (hl_psi_tagging['limb']=='lf')])
n2 = len(hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D2') &\
                            (hl_psi_tagging['limb']=='lf')])
#sns.set_theme(style="whitegrid")
f, ax1 = plt.subplots(1, 1, figsize=(3, 3), sharex=False)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=hl_psi_tagging, x="limb", y="r", hue="strain", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
#ax1.set_ylim([0,0.1])
ax1.get_legend().remove()
ax1.set_ylabel('Vector length', fontsize=16, family='arial')
ax1.set(xlabel=None)
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=16)
ax1.set_xticklabels(["LF", "LR", "RF", "RR"])
ax1.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')
ax1.set_title(group, fontsize=24, family='arial')
import matplotlib.lines as mlines
blue_line = mlines.Line2D([], [],ls='', color='blue', marker='.',
                          markersize=16, label=f'D1 MSN (n={n1})')
red_line = mlines.Line2D([], [],ls='', color='red', marker='.',
                          markersize=16, label=f'D2 MSN (n={n2})')

ax1.legend(handles=[blue_line, red_line], \
           loc='best', bbox_to_anchor=(1, 1), \
           labelcolor=['#0000FF','#FF0000'],\
        prop={'family': 'arial', "size": 16, 'stretch': 'normal'},
        frameon=False)
#plt.savefig(f'{output}d1-d2_vector_length_healthy.png', dpi=300, \
#            bbox_inches='tight', transparent=True)

# save data for graphpad prism
hl_r_lf_d1 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D1')&\
                                (hl_psi_tagging['limb']=='lf'),['r']].values
hl_r_lr_d1 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D1')&\
                                (hl_psi_tagging['limb']=='lr'),['r']].values
hl_r_rf_d1 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D1')&\
                                (hl_psi_tagging['limb']=='rf'),['r']].values
hl_r_rr_d1 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D1')&\
                                (hl_psi_tagging['limb']=='rr'),['r']].values
hl_r_lf_d2 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D2')&\
                                (hl_psi_tagging['limb']=='lf'),['r']].values
hl_r_lr_d2 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D2')&\
                                (hl_psi_tagging['limb']=='lr'),['r']].values
hl_r_rf_d2 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D2')&\
                                (hl_psi_tagging['limb']=='rf'),['r']].values
hl_r_rr_d2 = hl_psi_tagging.loc[(hl_psi_tagging['strain']=='D2')&\
                                (hl_psi_tagging['limb']=='rr'),['r']].values
hl_r_lf_d1_mean = np.mean(hl_r_lf_d1)
hl_r_lf_d1_sem = sem(hl_r_lf_d1)[0]
hl_r_lr_d1_mean = np.mean(hl_r_lr_d1)
hl_r_lr_d1_sem = sem(hl_r_lr_d1)[0]
hl_r_rf_d1_mean = np.mean(hl_r_rf_d1)
hl_r_rf_d1_sem = sem(hl_r_rf_d1)[0]
hl_r_rr_d1_mean = np.mean(hl_r_rr_d1)
hl_r_rr_d1_sem = sem(hl_r_rr_d1)[0]
hl_r_lf_d2_mean = np.mean(hl_r_lf_d2)
hl_r_lf_d2_sem = sem(hl_r_lf_d2)[0]
hl_r_lr_d2_mean = np.mean(hl_r_lr_d2)
hl_r_lr_d2_sem = sem(hl_r_lr_d2)[0]
hl_r_rf_d2_mean = np.mean(hl_r_rf_d2)
hl_r_rf_d2_sem = sem(hl_r_rf_d2)[0]
hl_r_rr_d2_mean = np.mean(hl_r_rr_d2)
hl_r_rr_d2_sem = sem(hl_r_rr_d2)[0]
hl_r_d1_mean = [hl_r_lf_d1_mean,hl_r_lr_d1_mean,hl_r_rf_d1_mean,hl_r_rr_d1_mean]
hl_r_d1_sem = [hl_r_lf_d1_sem,hl_r_lr_d1_sem,hl_r_rf_d1_sem,hl_r_rr_d1_sem]
hl_r_d2_mean = [hl_r_lf_d2_mean,hl_r_lr_d2_mean,hl_r_rf_d2_mean,hl_r_rr_d2_mean]
hl_r_d2_sem = [hl_r_lf_d2_sem,hl_r_lr_d2_sem,hl_r_rf_d2_sem,hl_r_rr_d2_sem]
hl_r = zip(hl_r_d1_mean,hl_r_d1_sem,hl_r_d2_mean,hl_r_d2_sem)
hl_r_df = pd.DataFrame(hl_r,columns=['d1-mean','d1-sem','d2-mean','d2-sem'])
#hl_r_df.to_csv(f'{output}all_mice_{group}_d1_d2_r.csv')

r_d1 = np.concatenate((hl_r_lf_d1,hl_r_lr_d1,hl_r_rf_d1,hl_r_rr_d1),axis=1)
r_d1 = np.transpose(r_d1)
r_d1_df = pd.DataFrame(r_d1)
#r_d1_df.to_csv(f'{output}all_mice_{group}_d1_r.csv')

r_d2 = np.concatenate((hl_r_lf_d2,hl_r_lr_d2,hl_r_rf_d2,hl_r_rr_d2),axis=1)
r_d2 = np.transpose(r_d2)
r_d2_df = pd.DataFrame(r_d2)
#r_d2_df.to_csv(f'{output}all_mice_{group}_d2_r.csv')

plt.savefig(f'{figpath}Figure4-E.pdf',dpi=300,bbox_inches='tight', transparent=True)

#%% ANOVA 2 way test on vector length
import statsmodels.api as sm
from statsmodels.formula.api import ols

#perform two-way ANOVA
model = ols('r ~ C(limb) + C(strain) + C(limb):C(strain)', data=hl_psi_tagging).fit()
sm.stats.anova_lm(model, typ=2)

#%% cross limb comparison
# we will use bioinfokit (v1.0.3 or later) for performing tukey HSD test
# check documentation here https://github.com/reneshbedre/bioinfokit
from bioinfokit.analys import stat
# perform multiple pairwise comparison (Tukey HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
res = stat()
# for main effect Genotype
res.tukey_hsd(df=hl_psi_tagging, res_var='r', xfac_var='strain', \
              anova_model='r~C(limb)+C(strain)+C(limb):C(strain)')
res.tukey_summary

#%% D1 vs D2 vector angle cross limbs - Figure 4G
f, ax2 = plt.subplots(1, 1, figsize=(3, 3), sharex=False)
group = 'HL'

lf_a_d1 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lf')&\
                   (psi_trimed['strain']=='D1')&\
                   (psi_trimed['tagging']==1),['a']].values
lf_m_d1, lf_sd_d1,_ = mf.circ_m(lf_a_d1)
lf_a_d2 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lf')&\
                   (psi_trimed['strain']=='D2')&\
                   (psi_trimed['tagging']==1),['a']].values
lf_m_d2, lf_sd_d2,_ = mf.circ_m(lf_a_d2)
lr_a_d1 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lr')&\
                   (psi_trimed['strain']=='D1')&\
                   (psi_trimed['tagging']==1),['a']].values
lr_m_d1, lr_sd_d1,_ = mf.circ_m(lr_a_d1)
lr_a_d2 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='lr')&\
                   (psi_trimed['strain']=='D2')&\
                   (psi_trimed['tagging']==1),['a']].values
lr_m_d2, lr_sd_d2,_ = mf.circ_m(lr_a_d2)
rf_a_d1 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rf')&\
                   (psi_trimed['strain']=='D1')&\
                   (psi_trimed['tagging']==1),['a']].values
rf_m_d1, rf_sd_d1,_ = mf.circ_m(rf_a_d1)
rf_a_d2 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rf')&\
                   (psi_trimed['strain']=='D2')&\
                   (psi_trimed['tagging']==1),['a']].values
rf_m_d2, rf_sd_d2,_ = mf.circ_m(rf_a_d2)
rr_a_d1 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rr')&\
                   (psi_trimed['strain']=='D1')&\
                   (psi_trimed['tagging']==1),['a']].values
rr_m_d1, rr_sd_d1,_ = mf.circ_m(rr_a_d1)
rr_a_d2 = psi_trimed.loc[(psi_trimed['group_id']==group)&\
                   (psi_trimed['limb']=='rr')&\
                   (psi_trimed['strain']=='D2')&\
                   (psi_trimed['tagging']==1),['a']].values
rr_m_d2, rr_sd_d2,_ = mf.circ_m(rr_a_d2)

mean_a_d1 = [lf_m_d1, lr_m_d1, rf_m_d1, rr_m_d1]
std_a_d1 = [lf_sd_d1, lr_sd_d1, rf_sd_d1, rr_sd_d1]
ind1 = np.arange(4)-0.03
mean_a_d2 = [lf_m_d2, lr_m_d2, rf_m_d2, rr_m_d2]
std_a_d2 = [lf_sd_d2, lr_sd_d2, rf_sd_d2, rr_sd_d2]
ind2 = np.arange(4)+0.03
ind = np.arange(4)
# permutation test
a_p_lf = mf.permut_angle(lf_a_d1, lf_a_d2, 1000)
a_p_lr = mf.permut_angle(lr_a_d1, lr_a_d2, 1000)
a_p_rf = mf.permut_angle(rf_a_d1, rf_a_d2, 1000)
a_p_rr = mf.permut_angle(rr_a_d1, rr_a_d2, 1000)

ax2.errorbar(ind1, mean_a_d1, linewidth=2, yerr=std_a_d1, fmt='none',\
             ecolor = '#0000FF', capsize = 4, capthick = 2)
ax2.scatter(ind1, mean_a_d1, s=60, c='#0000FF',label='D1')
ax2.errorbar(ind2, mean_a_d2, linewidth=2, yerr=std_a_d2, fmt='none',\
             ecolor = '#FF0000', capsize = 4, capthick = 2)  
ax2.scatter(ind2, mean_a_d2, s=60, c='#FF0000',label='D2')

ax2.legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
ax2.set_ylabel('Mean angle', fontsize=12, family='arial')
ax2.set(xlabel=None)
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=5, labelsize=12)
ax2.set_xticks(ind,("LF", "LR", "RF", "RR"))
#ax2.set_yticks([0, 0.5, 1, 1.5, 2])
ax2.set_yticks([0, 90, 180, 270, 360])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')
ax2.set_title(f'{group}', fontsize=12, family='arial')
plt.tight_layout()
# save data for graphpad prism
hl_a = zip(mean_a_d1,std_a_d1,mean_a_d2,std_a_d2)
hl_a_df = pd.DataFrame(hl_a,columns=['d1-mean','d1-sem','d2-mean','d2-sem'])
#hl_a_df.to_csv(f'{output}all_mice_{group}_d1_d2_a.csv')

plt.savefig(f'{figpath}Figure4-F.pdf',dpi=300,bbox_inches='tight', transparent=True)

#%% movement start encoding - d1 vs d2 - Figure 4I
#
from scipy.stats import ttest_ind
window = [-10, 10] # in second
binsize = 0.02 # bin size 20ms
bins = np.arange(window[0],window[1],binsize)
t = bins[:-1]
base_win = [-5,-1]
b_ind1 = int((base_win[0]-window[0])/binsize)
b_ind2 = int((base_win[1]-window[0])/binsize)
sig_win = [-0.5,0.5]
s_ind1 = int((sig_win[0]-window[0])/binsize)
s_ind2 = int((sig_win[1]-window[0])/binsize)
motion = 'start'
if motion=='start':
    psth_select = all_psth_start_trimed
elif motion=='stop':
    psth_select = all_psth_stop_trimed
else:
    print('please define your own selection')

group = 'HL'
select_psi = psi_trimed.loc[psi_trimed['limb']=='lr'].sort_values(by=['index'])
select_psi = select_psi.reset_index()
select_psi = select_psi.drop(columns=['level_0','index'])

d1_index = select_psi.loc[(select_psi['group_id']==group) &\
                          (select_psi['tagging']==1) &\
                          (select_psi['strain']=='D1')].index

d1_psth = psth_select[d1_index,:]
# normalized by pre-event mean firing rate
pre_event = np.mean(d1_psth[:,b_ind1:b_ind2],axis=1)
d1_psth = d1_psth/np.tile(np.expand_dims(pre_event,axis=1),d1_psth.shape[1])
d1_psth_mean = np.mean(d1_psth,axis=0)
d1_psth_sem = sem(d1_psth, axis=0)
d1_pre_single = np.mean(d1_psth[:,b_ind1:b_ind2],axis=1)
#d1_post_single = np.mean(d1_psth[:,s_ind1:s_ind2],axis=1)
#5/19/2023, Use maximum change
d1_post_single = np.max(np.abs(d1_psth[:,s_ind1:s_ind2]),axis=1)
d1_fract_single = (d1_post_single-d1_pre_single)/d1_pre_single

d2_index = select_psi.loc[(select_psi['group_id']==group) &\
                          (select_psi['tagging']==1) &\
                          (select_psi['strain']=='D2')].index

d2_psth = psth_select[d2_index,:]
# normalized by pre-event mean firing rate
pre_event = np.mean(d2_psth[:,b_ind1:b_ind2],axis=1)
d2_psth = d2_psth/np.tile(np.expand_dims(pre_event,axis=1),d2_psth.shape[1])
d2_psth_mean = np.mean(d2_psth,axis=0)
d2_psth_sem = sem(d2_psth, axis=0)
d2_pre_single = np.mean(d2_psth[:,b_ind1:b_ind2],axis=1)
#d2_post_single = np.mean(d2_psth[:,s_ind1:s_ind2],axis=1)
d2_post_single = np.max(np.abs(d2_psth[:,s_ind1:s_ind2]),axis=1)
d2_fract_single = (d2_post_single-d2_pre_single)/d2_pre_single

# save averaged data for graphpad prism
start_fr = zip(t.tolist(), d1_psth_mean.tolist(), d1_psth_sem.tolist(),\
               d2_psth_mean.tolist(), d2_psth_sem.tolist())
start_fr_df = pd.DataFrame(start_fr,\
                           columns=['time', 'd1-mean', 'd1-sem','d2-mean','d2-sem'])
#start_fr_df.to_csv(f'{output}fr2{motion}_{group}_d1_d2.csv')

#save single neuron data for graphpad prism
fract_single = {'D1':d1_fract_single, 'D2': d2_fract_single}
fract_single = dict([ (k,pd.Series(v)) for k,v in fract_single.items() ])
fract_single_df = pd.DataFrame(fract_single)
#d2_fract_single_df.to_csv(f'{output}fract2{motion}_single_{group}_d2.csv')
ttest_d1_d2 = ttest_ind(d1_fract_single, d2_fract_single, equal_var=False)

# plot figure
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.plot(t, d1_psth_mean, label='D1', color='#0000FF')
ax1.fill_between(t, d1_psth_mean+d1_psth_sem,\
                 d1_psth_mean-d1_psth_sem, \
                 color = '#0000FF', alpha=.3, linewidth=0)
ax1.plot(t, d2_psth_mean, label='D2', color='#FF0000')
ax1.fill_between(t, d2_psth_mean+d2_psth_sem,\
                 d2_psth_mean-d2_psth_sem, \
                 color='#FF0000', alpha=.3, linewidth=0)
ax1.set_xlabel(f'time from {motion} (s)', fontsize=24, family='arial')
ax1.set_ylabel('normalized firing rate (Hz)', fontsize=24, family='arial')
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend()
ax1.set_xlim([-1,1])
ax1.set_xticks([-1,-0.5,0,0.5,1])
ax1.tick_params(direction='out', width=1, length=5, labelsize=24)
#ax1.set_title(f'{group}-{motion}:{sig_win}, p: {ttest_d1_d2.pvalue}')

ax2 = fig.add_subplot(143)
fract_single_fig = {
    'stain': ["D1" for x in range(len(d1_fract_single))]+\
        ["D2" for x in range(len(d2_fract_single))],
    'fr': d1_fract_single.tolist() + d2_fract_single.tolist()}
sns.pointplot(data=fract_single_fig, x="stain", y="fr", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylabel('start modulation index', fontsize=24, family='arial')
ax2.set(xlabel=None)
ax2.set_ylim([0,1])
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=24)
ax2.set_xticklabels(["D1", "D2"])
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')
plt.tight_layout()

plt.savefig(f'{figpath}Figure4-G.pdf',dpi=300,bbox_inches='tight', transparent=True)
#%% movement speed encoding - d1 vs d2 - Figure 4J
all_binning_fr_trimed = zscore(all_binning_fr_trimed, axis=0)

group = 'HL'
select_psi = psi_trimed.loc[psi_trimed['limb']=='lf'].sort_values(by=['index'])
select_psi = select_psi.reset_index()
select_psi = select_psi.drop(columns=['level_0','index'])

d1_index = select_psi.loc[(select_psi['group_id']==group) &\
                          (select_psi['tagging']==1) &\
                          (select_psi['strain']=='D1')].index

d1_psth = all_binning_fr_trimed[:,d1_index]
d1_psth_mean = np.mean(d1_psth,axis=1)
d1_psth_sem = sem(d1_psth, axis=1)
d1_rval = np.abs(all_rval_trimed[d1_index])
d1_pval = all_rval_trimed[d1_index]
d1_rval_sig = np.abs(d1_rval[np.where(d1_pval<0.05)[0]])

d2_index = select_psi.loc[(select_psi['group_id']==group) &\
                          (select_psi['tagging']==1) &\
                          (select_psi['strain']=='D2')].index

d2_psth = all_binning_fr_trimed[:,d2_index]
d2_psth_mean = np.mean(d2_psth,axis=1)
d2_psth_sem = sem(d2_psth, axis=1)
d2_rval = np.abs(all_rval_trimed[d2_index])
d2_pval = all_rval_trimed[d2_index]
d2_rval_sig = np.abs(d2_rval[np.where(d2_pval<0.05)[0]])

#save single neuron data for graphpad prism
speed_single = {'D1':d1_rval, 'D2': d2_rval}
speed_single = dict([ (k,pd.Series(v)) for k,v in speed_single.items() ])
speed_single_df = pd.DataFrame(speed_single)
ttest_speed_d1_d2 = ttest_ind(d1_rval, d2_rval, equal_var=False)
# save data for graphpad prism
speed_fr = zip(binning_speed.tolist(), d1_psth_mean.tolist(),\
               d1_psth_sem.tolist(),d2_psth_mean.tolist(), d2_psth_sem.tolist())
speed_fr_df = pd.DataFrame(speed_fr,\
                           columns=['speed', 'd1-mean', 'd1-sem','d2-mean','d2-sem'])
#speed_fr_df.to_csv(f'{output}fr2speed_{group}_d1_d2.csv')

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.plot(binning_speed, d1_psth_mean, label='D1', color='#0000FF')
ax1.fill_between(binning_speed, d1_psth_mean+d1_psth_sem,\
                 d1_psth_mean-d1_psth_sem, \
                 color = '#0000FF', alpha=.3, linewidth=0)
ax1.plot(binning_speed, d2_psth_mean, label='D2', color='#FF0000')
ax1.fill_between(binning_speed, d2_psth_mean+d2_psth_sem,\
                 d2_psth_mean-d2_psth_sem, \
                 color='#FF0000', alpha=.3, linewidth=0)
ax1.set_xlabel('body speed (mm/s)', fontsize=24, family='arial')
ax1.set_ylabel('firing rate (z-scored)', fontsize=24, family='arial')
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.legend()
ax1.set_xlim([0,300])
ax1.set_xticks([0,100,200,300])
ax1.set_yticks([-2,-1,0,1,2])
ax1.tick_params(direction='out', width=2, length=5, labelsize=24)

ax2 = fig.add_subplot(143)
speed_single_fig = {
    'stain': ["D1" for x in range(len(d1_rval))]+\
        ["D2" for x in range(len(d2_rval))],
    'fr': d1_rval.tolist() + d2_rval.tolist()}
sns.pointplot(data=speed_single_fig, x="stain", y="fr", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylabel('speed encoding score', fontsize=24, family='arial')
ax2.set(xlabel=None)
ax2.set_ylim([0,1])
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=24)
ax2.set_xticklabels(["D1", "D2"])
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')
plt.tight_layout()

plt.savefig(f'{figpath}Figure4-H.pdf',dpi=300,bbox_inches='tight', transparent=True)

#%% pie plot of limb phase locking, speed encoding, start/stop encoding
#D1 & D2
#new Figure 4H 
from collections import Counter
from matplotlib_venn import venn3
strain = 'D2'
# phase locking encoding (at least one limb)
psi_p_hl_lf = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                             (psi_trimed['limb']=='lf')&\
                             (psi_trimed['strain']==strain)&\
                             (psi_trimed['tagging']==1),['p']].values
psi_p_hl_lr = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lr')&\
                           (psi_trimed['strain']==strain)&\
                           (psi_trimed['tagging']==1),['p']].values
psi_p_hl_rf = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='rf')&\
                           (psi_trimed['strain']==strain)&\
                           (psi_trimed['tagging']==1),['p']].values
psi_p_hl_rr = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='rr')&\
                           (psi_trimed['strain']==strain)&\
                           (psi_trimed['tagging']==1),['p']].values

psi_mtx = np.concatenate((psi_p_hl_lf,psi_p_hl_lr,psi_p_hl_rf,psi_p_hl_rr),axis=1)

psi_flag = np.zeros_like(psi_mtx)
psi_flag[np.where(psi_mtx<0.05)] = 1
psi_flag_sum = np.sum(psi_flag, axis=1)

psi_tag = np.zeros(psi_mtx.shape[0])
psi_tag[np.where(psi_flag_sum>=1)]=1

group_psi = np.where(psi_tag==1)[0]

# body speed encoding
all_pval = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf')&\
                           (psi_trimed['strain']==strain)&\
                           (psi_trimed['tagging']==1),['speed_score_p']].values
group_speed = np.where(all_pval<0.05)[0]

# start/stop encoding
all_start_p = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf')&\
                           (psi_trimed['strain']==strain)&\
                           (psi_trimed['tagging']==1),['start_score_p']].values
all_stop_p = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf')&\
                           (psi_trimed['strain']==strain)&\
                           (psi_trimed['tagging']==1),['stop_score_p']].values
start_stop = np.concatenate((np.expand_dims(all_start_p, axis=1), \
                            np.expand_dims(all_stop_p, axis=1)), axis=1)
start_stop_flag = np.zeros_like(start_stop)
start_stop_flag[np.where(start_stop<0.05)] = 1
start_stop_sum = np.sum(start_stop_flag, axis=1)
group_start = np.where(start_stop_sum>=1)[0]

dfA = pd.DataFrame(group_psi,columns=['A'])
dfB = pd.DataFrame(group_speed,columns=['B'])
dfC = pd.DataFrame(group_start,columns=['C'])

A = set(dfA.A)
B = set(dfB.B)
C = set(dfC.C)

AB_overlap = A & B  #compute intersection of set A & set B
AC_overlap = A & C
BC_overlap = B & C
ABC_overlap = A & B & C
A_rest = A - AB_overlap - AC_overlap #see left graphic
B_rest = B - AB_overlap - BC_overlap
C_rest = C - AC_overlap - BC_overlap
AB_only = AB_overlap - ABC_overlap   #see right graphic
AC_only = AC_overlap - ABC_overlap
BC_only = BC_overlap - ABC_overlap

sets = Counter()               #set order A, B, C   
sets['100'] = len(A_rest)      #100 denotes A on, B off, C off sets['010'] = len(B_rest)      #010 denotes A off, B on, C off
sets['010'] = len(B_rest)
sets['001'] = len(C_rest)      #001 denotes A off, B off, C on sets['110'] = len(AB_only)     #110 denotes A on, B on, C off
sets['110'] = len(AB_only)
sets['101'] = len(AC_only)     #101 denotes A on, B off, C on sets['011'] = len(BC_only)     #011 denotes A off, B on, C on sets['111'] = len(ABC_overlap) #011 denotes A on, B on, C onlabels = ('Group A', 'Group B', 'Group C')  
sets['011'] = len(BC_only)
sets['111'] = len(ABC_overlap)

labels = ('Phase', 'Speed', 'Start/Stop')

plt.figure(figsize=(1.5,1.5))
total = len(A.union(B.union(C)))

out = venn3(subsets=sets, \
            set_labels=labels,\
            set_colors=('red','green','blue'),alpha=0.5,\
            subset_label_formatter=lambda x: f"{(x/total):1.0%}")

for text in out.set_labels:
   text.set_fontsize(7)
for text in out.subset_labels:
    if text is not None:
        text.set_fontsize(7)  

plt.savefig(f'{figpath}Figure4H-HL-{strain}.pdf',dpi=300,bbox_inches='tight', transparent=True)