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
import numpy as np
import seaborn as sns

#%% Load single limb gaits in laser stimulation group
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, \
                            title='Select all-single-limb-gaits-laser-stim.pickle')
root.destroy()
print(filez)
with open(filez[0], 'rb') as f:
    gait = pickle.load(f)

#%% create path for figures
backslash_id = [i for i,x in enumerate(filez[0]) if x=='/']
output_path = filez[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% t test for pre/post laser

mouse_select = gait.loc[(gait['group_id']=='LA')]

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