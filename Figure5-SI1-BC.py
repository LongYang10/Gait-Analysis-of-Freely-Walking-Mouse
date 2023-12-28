# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:57:45 2023

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

import pickle
#%% Load single limb gait
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, \
                            title='Select all-single-limb-gaits.pickle')
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
#%%
from scipy.optimize import curve_fit
from itertools import zip_longest

mouse = 'F2234'
group = 'PD'
limb = 'lf'

gait_si = gait.loc[(gait['group_id']==group)&\
                   (gait['limb']==limb)&\
                   (gait['mouse_id']==mouse)]
stride_v = gait_si['stride_velocity']
stride_f = gait_si['cadence']
stride_length = gait_si['stride_length']
mean_v = gait_si['body_velocity']

fig, ax = plt.subplots(3,1,figsize=(6,18))

# calculate the linear fitting of speed data
coef = np.polyfit(stride_v,mean_v,1)
poly1d_fn = np.poly1d(coef)
x_data = np.linspace(0, 400, 437)
y_fit = poly1d_fn(x_data)
# plot stride speed vs body speed - panel-I
ax[2].plot(stride_v,mean_v, '.k', ms=6)
ax[2].plot(x_data, y_fit, ':r', lw=3)
ax[2].set_aspect('equal', adjustable='box')
ax[2].set_xlim([0,400])
ax[2].set_ylim([0,400])
ax[2].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[2].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[2].set_ylabel('body speed (mm/s)', fontsize=24, family='arial')
ax[2].spines[['top', 'right']].set_visible(False)
ax[2].spines['left'].set_linewidth(1)
ax[2].spines['bottom'].set_linewidth(1)

# save data for graphpad prism
sv_bv = zip(stride_v.tolist(), mean_v.tolist(), \
            x_data.tolist(), y_fit.tolist())
sv_bv_df = pd.DataFrame(sv_bv, columns=['stride speed', 'body speed',\
                                        'fit_t', 'fit_y'])
#sv_bv_df.to_csv(f'{output}stride_speed_body_speed_{mouse_id[-1]}.csv')

#%%
# inverse regression to fit the stride length to stride speed， 5/9/2023
x_data = np.arange(min(stride_v), 400, 1)
pars, cov = curve_fit(mf.inver_func, stride_v, stride_length)
y_fit = mf.inver_func(x_data, pars[0], pars[1])
# plot stride length vs stride speed - panel-G
ax[0].plot(stride_v, stride_length, '.k', ms=6)
ax[0].plot(x_data, y_fit, ':r', lw=3)
ax[0].set_box_aspect(1)
ax[0].set_xlim([0,400])
ax[0].set_ylim([0,100])
ax[0].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[0].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[0].set_ylabel('stride length (mm)', fontsize=24, family='arial')
ax[0].spines[['top', 'right']].set_visible(False)
ax[0].spines['left'].set_linewidth(1)
ax[0].spines['bottom'].set_linewidth(1)

# save data for graphpad prism
sl_sv = zip_longest(stride_v.tolist(), stride_length.tolist(), \
                    x_data.tolist(), y_fit.tolist(), fillvalue='-')
sl_sv_df = pd.DataFrame(sl_sv, columns=['stride speed', 'stride length',\
                                        'fit_t', 'fit_y'])
#sl_sv_df.to_csv(f'{output}stride_length_stride_speed_linear_{mouse_id[-1]}.csv')
#%%
# inverse regression to fit the stride frequency to stride speed， 5/9/2023
x_data = np.arange(min(stride_v), 400, 1)
pars, cov = curve_fit(mf.inver_func, stride_v, stride_f)
y_fit = mf.inver_func(x_data, pars[0], pars[1])

# plot stride frequency vs stride speed
ax[1].plot(stride_v, stride_f, '.k', ms=6)
ax[1].plot(x_data, y_fit, ':r', lw=3)
ax[1].set_box_aspect(1)
ax[1].set_xlim([0,400])
ax[1].set_ylim([0,8])
ax[1].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[1].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[1].set_ylabel('stride frequency ($s^{-1}$)', fontsize=24, family='arial')
ax[1].spines[['top', 'right']].set_visible(False)
ax[1].spines['left'].set_linewidth(1)
ax[1].spines['bottom'].set_linewidth(1)

# save data for graphpad prism
sf_sv = zip_longest(stride_v.tolist(), stride_f.tolist(), \
                    x_data.tolist(), y_fit.tolist(), fillvalue='-')
sf_sv_df = pd.DataFrame(sf_sv, columns=['stride speed', 'stride frequency',\
                                        'fit_t', 'fit_y'])

plt.tight_layout()

#plt.savefig(f'{figpath}Fig1-GHI-{group}-{limb}.pdf',dpi=300,bbox_inches='tight',transparent=True)

#%% plot distribution of gait parameters for 4 limbs
# new Figure 5 – Figure Supplement 1B,C
mouse = 'F2234'
group = 'PD'


gait_si_lf = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='lf')&\
                    (gait['mouse_id']==mouse)]
gait_si_lr = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='lr')&\
                    (gait['mouse_id']==mouse)]
gait_si_rf = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='rf')&\
                    (gait['mouse_id']==mouse)]
gait_si_rr = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='rr')&\
                    (gait['mouse_id']==mouse)]
# stride speed
stride_speed_lf = gait_si_lf['stride_velocity'].tolist()
stride_speed_lr = gait_si_lr['stride_velocity'].tolist()
stride_speed_rf = gait_si_rf['stride_velocity'].tolist()
stride_speed_rr = gait_si_rr['stride_velocity'].tolist()

lf_pd, lf_bins = np.histogram(stride_speed_lf, bins = 10, density=False, \
                               range=(0, 250))
lr_pd, _ = np.histogram(stride_speed_lr, bins = 10, density=False, \
                               range=(0, 250))
rf_pd, _ = np.histogram(stride_speed_rf, bins = 10, density=False, \
                               range=(0, 250))
rr_pd, _ = np.histogram(stride_speed_rr, bins = 10, density=False, \
                               range=(0, 250))
pd_x = lf_bins[:-1]+(lf_bins[1]-lf_bins[0])/2
# calculate the percentage of each bins
lf_pd = lf_pd/sum(lf_pd)*100
lr_pd = lr_pd/sum(lr_pd)*100
rf_pd = rf_pd/sum(rf_pd)*100
rr_pd = rr_pd/sum(rr_pd)*100
# save data for graphpad prism
pd_stride_speed = zip(pd_x, lf_pd, lr_pd, rf_pd, rr_pd)
pd_stride_speed_df = pd.DataFrame(pd_stride_speed, \
                              columns=['x','lf', 'lr', 'rf', 'rr'])
#pd_stride_speed_df.to_csv(f'{output}pd%_stride_speed-{mouse_id[-1]}.csv')

#stride length
stride_length_lf = gait_si_lf['stride_length'].tolist()
stride_length_lr = gait_si_lr['stride_length'].tolist()
stride_length_rf = gait_si_rf['stride_length'].tolist()
stride_length_rr = gait_si_rr['stride_length'].tolist()

lf_pd, lf_bins = np.histogram(stride_length_lf, bins = 10, density=False, \
         range=(0, 100))
lr_pd, _ = np.histogram(stride_length_lr, bins = 10, density=False, \
         range=(0, 100))
rf_pd, _ = np.histogram(stride_length_rf, bins = 10, density=False, \
         range=(0, 100))
rr_pd, _ = np.histogram(stride_length_rr, bins = 10, density=False, \
         range=(0, 100))
pd_x = lf_bins[:-1]+(lf_bins[1]-lf_bins[0])/2
lf_pd = lf_pd/sum(lf_pd)*100
lr_pd = lr_pd/sum(lr_pd)*100
rf_pd = rf_pd/sum(rf_pd)*100
rr_pd = rr_pd/sum(rr_pd)*100
# save data for graphpad prism
pd_stride_length = zip(pd_x, lf_pd, lr_pd, rf_pd, rr_pd)
pd_stride_length_df = pd.DataFrame(pd_stride_length, \
                              columns=['x','lf', 'lr', 'rf', 'rr'])
#pd_stride_length_df.to_csv(f'{output}pd%_stride_length-{mouse_id[-1]}.csv')

#stride frequency
stride_frequency_lf = gait_si_lf['cadence'].tolist()
stride_frequency_lr = gait_si_lr['cadence'].tolist()
stride_frequency_rf = gait_si_rf['cadence'].tolist()
stride_frequency_rr = gait_si_rr['cadence'].tolist()

lf_pd, lf_bins = np.histogram(stride_frequency_lf, bins = 10, density=False, \
         range=(0, 5))
lr_pd, _ = np.histogram(stride_frequency_lr, bins = 10, density=False, \
         range=(0, 5))
rf_pd, _ = np.histogram(stride_frequency_rf, bins = 10, density=False, \
         range=(0, 5))
rr_pd, _ = np.histogram(stride_frequency_rr, bins = 10, density=False, \
         range=(0, 5))
pd_x = lf_bins[:-1]+(lf_bins[1]-lf_bins[0])/2
lf_pd = lf_pd/sum(lf_pd)*100
lr_pd = lr_pd/sum(lr_pd)*100
rf_pd = rf_pd/sum(rf_pd)*100
rr_pd = rr_pd/sum(rr_pd)*100
# save data for graphpad prism
pd_stride_frequency = zip(pd_x, lf_pd, lr_pd, rf_pd, rr_pd)
pd_stride_frequency_df = pd.DataFrame(pd_stride_frequency, \
                              columns=['x','lf', 'lr', 'rf', 'rr'])
#pd_stride_frequency_df.to_csv(f'{output}pd%_stride_frequency-{mouse_id[-1]}.csv')

# plot the histogram
colors = ['#FF0000','#00FF00','#0000FF','#FF00FF']
fig, ax = plt.subplots(3,1,figsize=(6,18))
pd_stride_length_df.plot(ax=ax[0],x='x',color=colors)
pd_stride_frequency_df.plot(ax=ax[1],x='x',color=colors)
pd_stride_speed_df.plot(ax=ax[2],x='x',color=colors)

ax[0].set_box_aspect(1)
ax[0].set_xlim([0,100])
ax[0].set_ylim([0,50])
ax[0].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[0].set_xlabel('stride length (mm)', fontsize=24, family='arial')
ax[0].set_ylabel('% of stride', fontsize=24, family='arial')
ax[0].spines[['top', 'right']].set_visible(False)
ax[0].spines['left'].set_linewidth(1)
ax[0].spines['bottom'].set_linewidth(1)

ax[1].set_box_aspect(1)
ax[1].set_xlim([0,6])
ax[1].set_ylim([0,40])
ax[1].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[1].set_xlabel('stride frequency ($s^{-1}$)', fontsize=24, family='arial')
ax[1].set_ylabel('% of stride', fontsize=24, family='arial')
ax[1].spines[['top', 'right']].set_visible(False)
ax[1].spines['left'].set_linewidth(1)
ax[1].spines['bottom'].set_linewidth(1)

ax[2].set_box_aspect(1)
ax[2].set_xlim([0,250])
ax[2].set_ylim([0,30])
ax[2].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[2].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[2].set_ylabel('% of stride', fontsize=24, family='arial')
ax[2].spines[['top', 'right']].set_visible(False)
ax[2].spines['left'].set_linewidth(1)
ax[2].spines['bottom'].set_linewidth(1)

plt.tight_layout()
#plt.savefig(f'{figpath}Fig1-JKL.pdf',dpi=300,bbox_inches='tight',transparent=True)

#%%
from scipy.optimize import curve_fit
from itertools import zip_longest

mouse = 'F1590'
group = 'CT'
limb = 'lf'

gait_si = gait.loc[(gait['group_id']==group)&\
                   (gait['limb']==limb)&\
                   (gait['mouse_id']==mouse)]
stride_v = gait_si['stride_velocity']
stride_f = gait_si['cadence']
stride_length = gait_si['stride_length']
mean_v = gait_si['body_velocity']

fig, ax = plt.subplots(3,1,figsize=(6,18))

# calculate the linear fitting of speed data
coef = np.polyfit(stride_v,mean_v,1)
poly1d_fn = np.poly1d(coef)
x_data = np.linspace(0, 400, 437)
y_fit = poly1d_fn(x_data)
# plot stride speed vs body speed - panel-I
ax[2].plot(stride_v,mean_v, '.k', ms=6)
ax[2].plot(x_data, y_fit, ':r', lw=3)
ax[2].set_aspect('equal', adjustable='box')
ax[2].set_xlim([0,400])
ax[2].set_ylim([0,400])
ax[2].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[2].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[2].set_ylabel('body speed (mm/s)', fontsize=24, family='arial')
ax[2].spines[['top', 'right']].set_visible(False)
ax[2].spines['left'].set_linewidth(1)
ax[2].spines['bottom'].set_linewidth(1)

# save data for graphpad prism
sv_bv = zip(stride_v.tolist(), mean_v.tolist(), \
            x_data.tolist(), y_fit.tolist())
sv_bv_df = pd.DataFrame(sv_bv, columns=['stride speed', 'body speed',\
                                        'fit_t', 'fit_y'])
#sv_bv_df.to_csv(f'{output}stride_speed_body_speed_{mouse_id[-1]}.csv')

#%%
# inverse regression to fit the stride length to stride speed， 5/9/2023
x_data = np.arange(min(stride_v), 400, 1)
pars, cov = curve_fit(mf.inver_func, stride_v, stride_length)
y_fit = mf.inver_func(x_data, pars[0], pars[1])
# plot stride length vs stride speed - panel-G
ax[0].plot(stride_v, stride_length, '.k', ms=6)
ax[0].plot(x_data, y_fit, ':r', lw=3)
ax[0].set_box_aspect(1)
ax[0].set_xlim([0,400])
ax[0].set_ylim([0,100])
ax[0].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[0].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[0].set_ylabel('stride length (mm)', fontsize=24, family='arial')
ax[0].spines[['top', 'right']].set_visible(False)
ax[0].spines['left'].set_linewidth(1)
ax[0].spines['bottom'].set_linewidth(1)

# save data for graphpad prism
sl_sv = zip_longest(stride_v.tolist(), stride_length.tolist(), \
                    x_data.tolist(), y_fit.tolist(), fillvalue='-')
sl_sv_df = pd.DataFrame(sl_sv, columns=['stride speed', 'stride length',\
                                        'fit_t', 'fit_y'])
#sl_sv_df.to_csv(f'{output}stride_length_stride_speed_linear_{mouse_id[-1]}.csv')
#%%
# inverse regression to fit the stride frequency to stride speed， 5/9/2023
x_data = np.arange(min(stride_v), 400, 1)
pars, cov = curve_fit(mf.inver_func, stride_v, stride_f)
y_fit = mf.inver_func(x_data, pars[0], pars[1])

# plot stride frequency vs stride speed
ax[1].plot(stride_v, stride_f, '.k', ms=6)
ax[1].plot(x_data, y_fit, ':r', lw=3)
ax[1].set_box_aspect(1)
ax[1].set_xlim([0,400])
ax[1].set_ylim([0,8])
ax[1].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[1].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[1].set_ylabel('stride frequency ($s^{-1}$)', fontsize=24, family='arial')
ax[1].spines[['top', 'right']].set_visible(False)
ax[1].spines['left'].set_linewidth(1)
ax[1].spines['bottom'].set_linewidth(1)

# save data for graphpad prism
sf_sv = zip_longest(stride_v.tolist(), stride_f.tolist(), \
                    x_data.tolist(), y_fit.tolist(), fillvalue='-')
sf_sv_df = pd.DataFrame(sf_sv, columns=['stride speed', 'stride frequency',\
                                        'fit_t', 'fit_y'])

plt.tight_layout()

#plt.savefig(f'{figpath}Fig1-GHI-{group}-{limb}.pdf',dpi=300,bbox_inches='tight',transparent=True)

#%% plot distribution of gait parameters for 4 limbs
# new Figure 5 – Figure Supplement 1B,C
mouse = 'F1590'
group = 'CT'


gait_si_lf = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='lf')&\
                    (gait['mouse_id']==mouse)]
gait_si_lr = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='lr')&\
                    (gait['mouse_id']==mouse)]
gait_si_rf = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='rf')&\
                    (gait['mouse_id']==mouse)]
gait_si_rr = gait.loc[(gait['group_id']==group)&\
                    (gait['limb']=='rr')&\
                    (gait['mouse_id']==mouse)]
# stride speed
stride_speed_lf = gait_si_lf['stride_velocity'].tolist()
stride_speed_lr = gait_si_lr['stride_velocity'].tolist()
stride_speed_rf = gait_si_rf['stride_velocity'].tolist()
stride_speed_rr = gait_si_rr['stride_velocity'].tolist()

lf_pd, lf_bins = np.histogram(stride_speed_lf, bins = 10, density=False, \
                               range=(0, 450))
lr_pd, _ = np.histogram(stride_speed_lr, bins = 10, density=False, \
                               range=(0, 450))
rf_pd, _ = np.histogram(stride_speed_rf, bins = 10, density=False, \
                               range=(0, 450))
rr_pd, _ = np.histogram(stride_speed_rr, bins = 10, density=False, \
                               range=(0, 450))
pd_x = lf_bins[:-1]+(lf_bins[1]-lf_bins[0])/2
# calculate the percentage of each bins
lf_pd = lf_pd/sum(lf_pd)*100
lr_pd = lr_pd/sum(lr_pd)*100
rf_pd = rf_pd/sum(rf_pd)*100
rr_pd = rr_pd/sum(rr_pd)*100
# save data for graphpad prism
pd_stride_speed = zip(pd_x, lf_pd, lr_pd, rf_pd, rr_pd)
pd_stride_speed_df = pd.DataFrame(pd_stride_speed, \
                              columns=['x','lf', 'lr', 'rf', 'rr'])
#pd_stride_speed_df.to_csv(f'{output}pd%_stride_speed-{mouse_id[-1]}.csv')

#stride length
stride_length_lf = gait_si_lf['stride_length'].tolist()
stride_length_lr = gait_si_lr['stride_length'].tolist()
stride_length_rf = gait_si_rf['stride_length'].tolist()
stride_length_rr = gait_si_rr['stride_length'].tolist()

lf_pd, lf_bins = np.histogram(stride_length_lf, bins = 10, density=False, \
         range=(0, 100))
lr_pd, _ = np.histogram(stride_length_lr, bins = 10, density=False, \
         range=(0, 100))
rf_pd, _ = np.histogram(stride_length_rf, bins = 10, density=False, \
         range=(0, 100))
rr_pd, _ = np.histogram(stride_length_rr, bins = 10, density=False, \
         range=(0, 100))
pd_x = lf_bins[:-1]+(lf_bins[1]-lf_bins[0])/2
lf_pd = lf_pd/sum(lf_pd)*100
lr_pd = lr_pd/sum(lr_pd)*100
rf_pd = rf_pd/sum(rf_pd)*100
rr_pd = rr_pd/sum(rr_pd)*100
# save data for graphpad prism
pd_stride_length = zip(pd_x, lf_pd, lr_pd, rf_pd, rr_pd)
pd_stride_length_df = pd.DataFrame(pd_stride_length, \
                              columns=['x','lf', 'lr', 'rf', 'rr'])
#pd_stride_length_df.to_csv(f'{output}pd%_stride_length-{mouse_id[-1]}.csv')

#stride frequency
stride_frequency_lf = gait_si_lf['cadence'].tolist()
stride_frequency_lr = gait_si_lr['cadence'].tolist()
stride_frequency_rf = gait_si_rf['cadence'].tolist()
stride_frequency_rr = gait_si_rr['cadence'].tolist()

lf_pd, lf_bins = np.histogram(stride_frequency_lf, bins = 10, density=False, \
         range=(1, 6))
lr_pd, _ = np.histogram(stride_frequency_lr, bins = 10, density=False, \
         range=(1, 6))
rf_pd, _ = np.histogram(stride_frequency_rf, bins = 10, density=False, \
         range=(1, 6))
rr_pd, _ = np.histogram(stride_frequency_rr, bins = 10, density=False, \
         range=(1, 6))
pd_x = lf_bins[:-1]+(lf_bins[1]-lf_bins[0])/2
lf_pd = lf_pd/sum(lf_pd)*100
lr_pd = lr_pd/sum(lr_pd)*100
rf_pd = rf_pd/sum(rf_pd)*100
rr_pd = rr_pd/sum(rr_pd)*100
# save data for graphpad prism
pd_stride_frequency = zip(pd_x, lf_pd, lr_pd, rf_pd, rr_pd)
pd_stride_frequency_df = pd.DataFrame(pd_stride_frequency, \
                              columns=['x','lf', 'lr', 'rf', 'rr'])
#pd_stride_frequency_df.to_csv(f'{output}pd%_stride_frequency-{mouse_id[-1]}.csv')

# plot the histogram
colors = ['#FF0000','#00FF00','#0000FF','#FF00FF']
fig, ax = plt.subplots(3,1,figsize=(6,18))
pd_stride_length_df.plot(ax=ax[0],x='x',color=colors)
pd_stride_frequency_df.plot(ax=ax[1],x='x',color=colors)
pd_stride_speed_df.plot(ax=ax[2],x='x',color=colors)

ax[0].set_box_aspect(1)
ax[0].set_xlim([0,100])
ax[0].set_ylim([0,50])
ax[0].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[0].set_xlabel('stride length (mm)', fontsize=24, family='arial')
ax[0].set_ylabel('% of stride', fontsize=24, family='arial')
ax[0].spines[['top', 'right']].set_visible(False)
ax[0].spines['left'].set_linewidth(1)
ax[0].spines['bottom'].set_linewidth(1)

ax[1].set_box_aspect(1)
ax[1].set_xlim([0,6])
ax[1].set_ylim([0,40])
ax[1].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[1].set_xlabel('stride frequency ($s^{-1}$)', fontsize=24, family='arial')
ax[1].set_ylabel('% of stride', fontsize=24, family='arial')
ax[1].spines[['top', 'right']].set_visible(False)
ax[1].spines['left'].set_linewidth(1)
ax[1].spines['bottom'].set_linewidth(1)

ax[2].set_box_aspect(1)
ax[2].set_xlim([0,450])
ax[2].set_ylim([0,30])
ax[2].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[2].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[2].set_ylabel('% of stride', fontsize=24, family='arial')
ax[2].spines[['top', 'right']].set_visible(False)
ax[2].spines['left'].set_linewidth(1)
ax[2].spines['bottom'].set_linewidth(1)

plt.tight_layout()
#plt.savefig(f'{figpath}Fig1-JKL.pdf',dpi=300,bbox_inches='tight',transparent=True)



