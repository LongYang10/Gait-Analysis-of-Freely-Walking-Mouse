# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:31:16 2023
@author: lonya
"""
#%% preset
import tkinter as tk
import tkinter.filedialog as fd
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import my_funcs as mf
import seaborn as sns
#%% Load all single animal gaits
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, \
                            title='Choose all-single-animal-gaits.pickle')
root.destroy()
print(filez)
with open(filez[0], 'rb') as f:
    mouse_mean_d = pickle.load(f)
#%% create path for figures
backslash_id = [i for i,x in enumerate(filez[0]) if x=='/']
output_path = filez[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42  
#%% comparison CT vs PD - Figure 5CDEF
mouse_select = mouse_mean_d.loc[(mouse_mean_d['group_id']=='CT')|\
                                (mouse_mean_d['group_id']=='PD')]
fig = plt.figure(figsize=(8,12))

ax1 = fig.add_subplot(221)
sns.pointplot(data=mouse_select, x="group_id", y="head_angle", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
ax1.set_ylabel('mean change in heading (deg)', fontsize=24, family='arial')
ax1.set(xlabel=None)
ax1.set_ylim([-60,20])
ax1.spines[['left','bottom']].set_linewidth(2)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=2, length=8, labelsize=24)
ax1.set_xticklabels(["sham", "6OHDA"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')

# significance test
hd_a_ct = mouse_mean_d.loc[mouse_mean_d['group_id']=='CT',\
                               ['head_angle']].values
hd_a_pd = mouse_mean_d.loc[mouse_mean_d['group_id']=='PD',\
                               ['head_angle']].values
hd_m_ct, hd_sd_ct,_ = mf.circ_m(hd_a_ct)
hd_m_pd, hd_sd_pd,_ = mf.circ_m(hd_a_pd)
hd_p = mf.permut_angle(hd_a_ct, hd_a_pd, 1000)
# save data for graphpad prism
hd_a = [[hd_m_ct, hd_sd_ct, hd_m_pd, hd_sd_pd]]
hd_a_df = pd.DataFrame(hd_a,columns=['sham-mean','sham-sem','6OHDA-mean','6OHDA-sem'])
#hd_a_df.to_csv(f'{output}head_angle_CT_PD.csv')

### mean walking speed comparison CT vs PD - Figure 5D
ax2 = fig.add_subplot(222)
sns.pointplot(data=mouse_select, x="group_id", y="walk_speed", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylabel('walking body speed (mm/s)', fontsize=24, family='arial')
ax2.set(xlabel=None)
ax2.set_ylim([0,200])
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=24)
ax2.set_xticklabels(["sham", "6OHDA"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')

### total travel distance comparison CT vs PD - Figure 5E
ax3 = fig.add_subplot(223)
sns.pointplot(data=mouse_select, x="group_id", y="walk_distance", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax3)
ax3.set_ylabel('total distance traveled (mm)', fontsize=24, family='arial')
ax3.set(xlabel=None)
ax3.set_ylim([0,50000])
ax3.spines[['left','bottom']].set_linewidth(2)
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(direction='out', width=2, length=8, labelsize=24)
ax3.set_xticklabels(["sham", "6OHDA"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(ax3.get_yticks(),family='arial')

### initiation rate comparison CT vs PD - Figure 5E
ax4 = fig.add_subplot(224)
sns.pointplot(data=mouse_select, x="group_id", y="walk_rate", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax4)
ax4.set_ylabel('initiation rate (starts/min)', fontsize=24, family='arial')
ax4.set(xlabel=None)
ax4.set_ylim([0,5])
ax4.spines[['left','bottom']].set_linewidth(2)
ax4.spines[['top', 'right']].set_visible(False)
ax4.tick_params(direction='out', width=2, length=8, labelsize=24)
ax4.set_xticklabels(["sham", "6OHDA"],rotation=45)
#ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax4.set_yticklabels(ax4.get_yticks(),family='arial')

plt.tight_layout()

plt.savefig(f'{figpath}Figure5-CDEF.pdf',dpi=300,bbox_inches='tight', transparent=True)

#%% mean gaits comparison CT vs PD - Figure 5GHIJ
#prepare dataframe
new_mouse_select = {
    'group_id': mouse_select['group_id'].values.tolist()+\
                mouse_select['group_id'].values.tolist()+\
                mouse_select['group_id'].values.tolist()+\
                mouse_select['group_id'].values.tolist(),
    'limb':["lf" for x in range(len(mouse_select.index))]+\
           ["lr" for x in range(len(mouse_select.index))]+\
           ["rf" for x in range(len(mouse_select.index))]+\
           ["rr" for x in range(len(mouse_select.index))],
    'stride_length': mouse_select['stride_length_lf'].values.tolist()+\
                     mouse_select['stride_length_lr'].values.tolist()+\
                     mouse_select['stride_length_rf'].values.tolist()+\
                     mouse_select['stride_length_rr'].values.tolist(),
    'stride_dur':mouse_select['stride_dur_lf'].values.tolist()+\
                 mouse_select['stride_dur_lr'].values.tolist()+\
                 mouse_select['stride_dur_rf'].values.tolist()+\
                 mouse_select['stride_dur_rr'].values.tolist(),
    'stride_speed': mouse_select['stride_speed_lf'].values.tolist()+\
                    mouse_select['stride_speed_lr'].values.tolist()+\
                    mouse_select['stride_speed_rf'].values.tolist()+\
                    mouse_select['stride_speed_rr'].values.tolist(),
    'sw_st_ratio':mouse_select['sw_st_ratio_lf'].values.tolist()+\
                  mouse_select['sw_st_ratio_lr'].values.tolist()+\
                  mouse_select['sw_st_ratio_rf'].values.tolist()+\
                  mouse_select['sw_st_ratio_rr'].values.tolist(),
    'stride_length_cv':mouse_select['stride_length_cv_lf'].values.tolist()+\
                      mouse_select['stride_length_cv_lr'].values.tolist()+\
                      mouse_select['stride_length_cv_rf'].values.tolist()+\
                      mouse_select['stride_length_cv_rr'].values.tolist(),
    'stride_dur_cv':mouse_select['stride_dur_cv_lf'].values.tolist()+\
                    mouse_select['stride_dur_cv_lr'].values.tolist()+\
                    mouse_select['stride_dur_cv_rf'].values.tolist()+\
                    mouse_select['stride_dur_cv_rr'].values.tolist(),
    'stride_speed_cv': mouse_select['stride_speed_cv_lf'].values.tolist()+\
                       mouse_select['stride_speed_cv_lr'].values.tolist()+\
                       mouse_select['stride_speed_cv_rf'].values.tolist()+\
                       mouse_select['stride_speed_cv_rr'].values.tolist(),
    'sw_st_ratio_cv':mouse_select['sw_st_ratio_cv_lf'].values.tolist()+\
                  mouse_select['sw_st_ratio_cv_lr'].values.tolist()+\
                  mouse_select['sw_st_ratio_cv_rf'].values.tolist()+\
                  mouse_select['sw_st_ratio_cv_rr'].values.tolist()
                  }
mouse_select_new = pd.DataFrame(data=new_mouse_select)

fig = plt.figure(figsize=(12,8))

### plot stride length - Figure 5G
ax1 = fig.add_subplot(221)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=mouse_select_new, x="limb", y="stride_length", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax1)
ax1.set_ylim([35,55])
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
sns.pointplot(data=mouse_select_new, x="limb", y="stride_dur", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax2)
ax2.set_ylim([0.25,0.45])
ax2.get_legend().remove()
ax2.set_ylabel('mean stride duration (s)', fontsize=16, family='arial')
ax2.set(xlabel=None)
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(direction='out', width=2, length=8, labelsize=16)
ax2.set_xticklabels(["LF", "LR", "RF", "RR"])
ax2.set_yticks([0.25, 0.3, 0.35, 0.4, 0.45])
ax2.set_yticklabels(ax2.get_yticks(),family='arial')

### plot stride speed - Figure 5I
ax3 = fig.add_subplot(223)
# Draw a pointplot to show pulse as a function of three categorical factors
sns.pointplot(data=mouse_select_new, x="limb", y="stride_speed", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax3)
ax3.set_ylim([50,200])
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
sns.pointplot(data=mouse_select_new, x="limb", y="sw_st_ratio", hue="group_id", \
              palette=['#0000FF','#FF0000'], join=False, errorbar="se", \
              scale=1, capsize=0.1, dodge=True, errwidth=2, ax=ax4)
ax4.set_ylim([0.5,0.9])
ax4.get_legend().remove()
ax4.set_ylabel('mean swing:stance duration', fontsize=16, family='arial')
ax4.set(xlabel=None)
ax4.spines[['left','bottom']].set_linewidth(2)
ax4.spines[['top', 'right']].set_visible(False)
ax4.tick_params(direction='out', width=2, length=8, labelsize=16)
ax4.set_xticklabels(["LF", "LR", "RF", "RR"])
ax4.set_yticks([0.6,0.8])
ax4.set_yticklabels(ax4.get_yticks(),family='arial')

plt.savefig(f'{figpath}Figure5-GHIJ.pdf',dpi=300,bbox_inches='tight', transparent=True)
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

plt.savefig(f'{figpath}Figure S2-CDEF.pdf',dpi=300,bbox_inches='tight', transparent=True)