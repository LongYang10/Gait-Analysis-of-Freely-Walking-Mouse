# -*- coding: utf-8 -*-
"""
Analysis script for Long et al paper: 
"Striatal Neuron Phase-Locking to the Gait Cycle during Locomotion"

Code by Long Yang
Updated August 2023
"""
#%% preset
from tkinter import Tk
import tkinter.filedialog as fd
from tkinter import messagebox
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from itertools import zip_longest
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
#%% Load Data (F2203)
root = Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Select F2203.mat files',\
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
figpath = str(output_path) + 'Figure 1/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% processing data
mouse_gait = pd.DataFrame()
    
matfile = sio.loadmat(all_files[0], squeeze_me = True)
eMouse = matfile['eMouse']
# get locomotion trajectory
mouse2D = mf.mouse2D_format(eMouse)
walkTimes = mouse2D['walkTimes']
params = mouse2D['params']

# calculate self-view mouse, gait cycle
svm = mf.createSVM(mouse2D)
svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')

# get gaits & strides
gait, stride = mf.calculate_gait(svm_bf, mouse2D) #1/11/2023
gait_trimed, stride_trimed = mf.trim_gait(gait, stride)

# get kinematics
kinematics = mf.mouse_kinematics(mouse2D, params)

# get full stride
full_stride, _ = mf.get_full_stride(stride, walkTimes)

# get limb phase angle
#svm_angle = mf.get_mouse_angle(svm_bf)
svm_angle = mf.get_mouse_angle_pct(svm_bf, stride_trimed)

#create dataframe
temp_gait = mf.get_gait_df(gait_trimed, 'F2203', 'HL', 'D1', 'tagging')
mouse_gait = pd.concat([mouse_gait, temp_gait], ignore_index = True)
#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
    
#%% plot whole body speed - label single walking bout - Figure1D
# F2203 - walking bout 24/27
# index 24: low speed walking bout
# index 27: high speed walking bout
walk_id = 27 # index of the illustrated high speed walking bout
wk = walkTimes[walk_id,:]
wk_24 = walkTimes[24,:]
dt = wk-wk_24
speed_sm = gaussian_filter(kinematics['bd_speed'], sigma = 10)
t = np.arange(wk[0], wk[1], 1/params['fr'])
index = np.arange(int(np.round(wk[0]*params['fr'])), \
                  int(np.round(wk[1]*params['fr'])), 1)
N = min(t.shape,index.shape)[0]
t = t[:N]-t[0]
index = index[:N]
ex_win = 30 #5min pre/post
wk_ex = [max(wk[0]-ex_win, 0), min(wk[1]+ex_win, params['rec_duration'])]
t_ex = np.arange(wk_ex[0], wk_ex[1], 1/params['fr'])
index_ex = np.arange(int(np.round(wk_ex[0]*params['fr'])), \
                  int(np.round(wk_ex[1]*params['fr'])), 1)
N = min(t_ex.shape,index_ex.shape)[0]
t_ex = t_ex[:N]-t_ex[0]
index_ex = index_ex[:N]

fig = plt.figure(figsize=(7,4))
# plot phase-FR
ax0 = fig.add_subplot(211)
ax0.plot(t_ex, speed_sm[index_ex], color = '0', lw=1)
ax0.axvline(x=ex_win, color='b', lw=1, ls='--')
ax0.axvline(x=(t_ex[-1]-ex_win), color='b', lw=1, ls='--')
# special v-line for nearby low speed walking bout - 24
ax0.axvline(x = ex_win-dt[0], color='r', lw=1, ls='--')
ax0.axvline(x=(t_ex[-1]-ex_win-dt[1]), color='r', lw=1, ls='--')
ax0.spines[['top', 'right', 'bottom']].set_visible(False)
ax0.spines['left'].set_linewidth(1)
ax0.set_yticks([0,100,200,300])
ax0.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax0.set_ylabel('body speed (mm/s)', fontsize=7, family='arial')
ax0.tick_params(direction='out', width=1, length=5, labelsize=7)

fontprops = fm.FontProperties(family='arial',size=7)
scalebar = AnchoredSizeBar(ax0.transData,
                           10, '10 s', 
                           loc = 'lower right', 
                           bbox_to_anchor=(0.9, -0.2),
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1,
                           bbox_transform=ax0.transAxes,
                           fontproperties=fontprops)

ax0.add_artist(scalebar)

### plot low speed walking bout
ax1 = fig.add_subplot(223)
walk_id = 24 # change the index (24,27) to plot high/low speed cycle
wk = walkTimes[walk_id,:]
t = np.arange(wk[0], wk[1], 1/params['fr'])
index = np.arange(int(np.round(wk[0]*params['fr'])), \
                  int(np.round(wk[1]*params['fr'])), 1)
N = min(t.shape,index.shape)[0]
t = t[:N]-t[0]
index = index[:N]
# take 1s data
t = t[t.shape[0]-81:-1]
t = t-t[0]
index = index[index.shape[0]-81:-1]
ax1.plot(t, svm_bf['lf'][index], lw=1, color = '#FF0000')
ax1.plot(t, svm_bf['lr'][index], lw=1, color = '#00FF00')
ax1.plot(t, svm_bf['rf'][index], lw=1, color = '#0000FF')
ax1.plot(t, svm_bf['rr'][index], lw=1, color = '#FF00FF')
ax1.set_ylim([-16, 25])
ax1.set_xlim([0, 1])
ax1.spines[['top', 'right', 'bottom']].set_visible(False)
ax1.spines['left'].set_linewidth(1)
ax1.set_yticks([-20,-10,0,10,20])
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax1.set_ylabel('position (mm)', fontsize=7, family='arial')
ax1.tick_params(direction='out', width=1, length=5, labelsize=7)
fontprops = fm.FontProperties(family='arial',size=7)
scalebar = AnchoredSizeBar(ax1.transData,
                           0.2, '0.2 s', 
                           loc = 'lower right', 
                           bbox_to_anchor=(0.9, -0.2),
                           bbox_transform=ax1.transAxes,
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1,
                           fontproperties=fontprops)

ax1.add_artist(scalebar)

### plot high speed walking bout
ax2 = fig.add_subplot(224)
walk_id = 27 # change the index (24,27) to plot high/low speed cycle
wk = walkTimes[walk_id,:]

t = np.arange(wk[0], wk[1], 1/params['fr'])
index = np.arange(int(np.round(wk[0]*params['fr'])), \
                  int(np.round(wk[1]*params['fr'])), 1)
N = min(t.shape,index.shape)[0]
t = t[:N]-t[0]
index = index[:N]

# take 1s data
t = t[t.shape[0]-81:-1]
t = t-t[0]
index = index[index.shape[0]-81:-1]
ax2.plot(t, svm_bf['lf'][index], lw=1, color = '#FF0000')
ax2.plot(t, svm_bf['lr'][index], lw=1, color = '#00FF00')
ax2.plot(t, svm_bf['rf'][index], lw=1, color = '#0000FF')
ax2.plot(t, svm_bf['rr'][index], lw=1, color = '#FF00FF')
ax2.set_ylim([-16, 25])
ax2.set_xlim([0, 1])
ax2.spines[['top', 'right', 'bottom']].set_visible(False)
ax2.spines['left'].set_linewidth(1)
ax2.set_yticks([-20,-10,0,10,20])
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax2.set_ylabel('position (mm)', fontsize=7, family='arial')
ax2.tick_params(direction='out', width=1, length=5, labelsize=7)
fontprops = fm.FontProperties(family='arial',size=7)
scalebar = AnchoredSizeBar(ax2.transData,
                           0.2, '0.2 s', 
                           loc = 'lower right', 
                           bbox_to_anchor=(0.9, -0.2),
                           bbox_transform=ax2.transAxes,
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1,
                           fontproperties=fontprops)

ax2.add_artist(scalebar)
plt.tight_layout()
plt.savefig(f'{figpath}Fig1-D.pdf', dpi=300, bbox_inches='tight', transparent=True)

#%% plot the whole body trajectory - Figure1C
f, ax1 = plt.subplots(1, 1, figsize=(1.8, 1.8))
bd = mouse2D['body']
camTimes = params['camTimes']

ind1 = int(camTimes[0]*params['fr'])+3
ind2 = int(camTimes[1]*params['fr'])

walk_df = pd.DataFrame()
for i in range(int(walkTimes.shape[0])):
    ind1 = int(walkTimes[i,0]*params['fr'])
    ind2 = int(walkTimes[i,1]*params['fr'])
    color = range(ind2-ind1)
    ax1.scatter(bd[ind1:ind2,0], bd[ind1:ind2,1], s=0.1,c=color,cmap= 'Greys')
    temp_walk = pd.DataFrame(bd[ind1:ind2,:])
    walk_df = pd.concat([walk_df, temp_walk], ignore_index = True)
# save data for graphpad prism
walk_df.columns = ['x', 'y']
#walk_df.to_csv(f'{output}walk_trajectory_{mouse_id[-1]}.csv')
ax1.set_xlim([0, 630])
ax1.set_ylim([0, 630])
ax1.set_xlabel('x position (mm)', fontsize=7, family='arial')
ax1.set_ylabel('y position (mm)', fontsize=7, family='arial')
ax1.tick_params(direction='out', width=0.5, length=4, labelsize=7)
ax1.set_xticks([0, 200, 400, 600])
ax1.set_xticklabels(ax1.get_xticks(), family='arial')
ax1.set_yticks([0, 200, 400, 600])
ax1.set_yticklabels(ax1.get_yticks(), family='arial')
ax1.set_aspect('equal', adjustable='box')
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['top'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)

plt.savefig(f'{figpath}Fig1-C.pdf',dpi=300,bbox_inches='tight',transparent=True)
#%% plot time coordination - Figure1E,F
temp_stride = np.asarray(full_stride)
temp_stride = np.reshape(temp_stride, (int(len(full_stride)/12),12))

lr_swing = temp_stride[:,1]
lf_swing = temp_stride[:,4]
rf_swing = temp_stride[:,7]
rr_swing = temp_stride[:,10]

lr_lr = lr_swing-lr_swing
lf_lr = lf_swing-lr_swing
rf_lr = rf_swing-lr_swing
rr_lr = rr_swing-lr_swing

# create dataframe for figure
swing2lr_df_fig = {
    'limb': ["lf" for x in range(len(lf_lr))]+\
            ["lr" for x in range(len(lr_lr))]+\
            ["rf" for x in range(len(rf_lr))]+\
            ["rr" for x in range(len(rr_lr))],
    'swt': lf_lr.tolist()+lr_lr.tolist()+rf_lr.tolist()+rr_lr.tolist()}
# plot figure
fig = plt.figure(figsize=(6,4))
ax0 = fig.add_subplot(121)
sns.pointplot(data=swing2lr_df_fig, x="limb", y="swt", \
              palette=['#000000'], join=False, errorbar="sd", \
              scale=1, capsize=0.2, dodge=True, errwidth=2, ax=ax0)
#ax0.set_ylim([-,0.1])
ax0.set_ylabel('swing time relative to LR (s)', fontsize=16, family='arial')
ax0.set(xlabel=None)
ax0.spines[['left','bottom']].set_linewidth(1)
ax0.spines[['top', 'right']].set_visible(False)
ax0.tick_params(direction='out', width=1, length=3, labelsize=16)
ax0.set_xticklabels(["LF", "LR", "RF", "RR"])
ax0.set_yticks([0, 0.1, 0.2,0.3])
ax0.set_yticklabels(ax0.get_yticks(),family='arial')

###plot phase coordination - Figure1F
temp_stride = np.asarray(full_stride)
temp_stride = np.reshape(temp_stride, (int(len(full_stride)/12),12))

lr_stance = temp_stride[:,0]
lf_stance = temp_stride[:,3]
rf_stance = temp_stride[:,6]
rr_stance = temp_stride[:,9]

# find the corresponding phase angle at each stance time, reference to lr
lr_index = np.round(lr_stance*params['fr']).astype(int)
lr_stance_p = svm_angle['lr'][lr_index]
lf_index = np.round(lf_stance*params['fr']).astype(int)
lf_stance_p = svm_angle['lr'][lf_index]
rf_index = np.round(rf_stance*params['fr']).astype(int)
rf_stance_p = svm_angle['lr'][rf_index]
rr_index = np.round(rr_stance*params['fr']).astype(int)
rr_stance_p = svm_angle['lr'][rr_index]

lf_m, lf_sd = mf.circ_m(lf_stance_p)
lr_m, lr_sd = mf.circ_m(lr_stance_p)
rf_m, rf_sd = mf.circ_m(rf_stance_p)
rr_m, rr_sd = mf.circ_m(rr_stance_p)

mean_a = [lf_m, rf_m, rr_m]
std_a = [lf_sd, rf_sd, rr_sd]
ind = np.arange(3)
# create dataframe for figure
ax1 = fig.add_subplot(122)
ax1.errorbar(ind, mean_a, linewidth=2, yerr=std_a, fmt='none',\
             ecolor = '#000000', capsize = 4, capthick = 2)
ax1.scatter(ind, mean_a, s=60, c='#000000')
ax1.set_ylabel('limb phase relative to LR (deg)', fontsize=16, family='arial')
ax1.set(xlabel=None)
ax1.spines[['left','bottom']].set_linewidth(1)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(direction='out', width=1, length=3, labelsize=16)
ax1.set_xticks(ind,("LF", "RF", "RR"))
ax1.set_yticks([0, 90, 180, 270, 360])
ax1.set_yticklabels(ax1.get_yticks(),family='arial')
plt.tight_layout()
plt.savefig(f'{figpath}Fig1-EF.pdf',dpi=300,bbox_inches='tight',transparent=True)
#%% fig - gait parameters Figure1-G,H,I
stride_v = gait_trimed['lf']['stride_velocity']
stride_f = gait_trimed['lf']['cadence']
stride_length = gait_trimed['lf']['stride_length']
stride = stride_trimed['lf']['stride']
speed = kinematics['bd_speed']
mean_v = np.zeros_like(stride_v)

# find the corresponding whole body speed in each stride
for i in range(stride.shape[0]):
    ind1 = int(np.round(stride[i,0]*params['fr']))
    ind2 = int(np.round(stride[i,2]*params['fr']))
    mean_v[i] = np.mean(speed[ind1:ind2])

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

plt.tight_layout()

plt.savefig(f'{figpath}Fig1-GHI.pdf',dpi=300,bbox_inches='tight',transparent=True)
#%% plot distribution of gait parameters for 4 limbs, Figure1-J,K,L

# stride speed
stride_speed_lf = gait_trimed['lf']['stride_velocity'].tolist()
stride_speed_lr = gait_trimed['lr']['stride_velocity'].tolist()
stride_speed_rf = gait_trimed['rf']['stride_velocity'].tolist()
stride_speed_rr = gait_trimed['rr']['stride_velocity'].tolist()

lf_pd, lf_bins = np.histogram(stride_speed_lf, bins = 10, density=False, \
                               range=(50, 350))
lr_pd, _ = np.histogram(stride_speed_lr, bins = 10, density=False, \
                               range=(50, 350))
rf_pd, _ = np.histogram(stride_speed_rf, bins = 10, density=False, \
                               range=(50, 350))
rr_pd, _ = np.histogram(stride_speed_rr, bins = 10, density=False, \
                               range=(50, 350))
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
stride_length_lf = gait_trimed['lf']['stride_length'].tolist()
stride_length_lr = gait_trimed['lr']['stride_length'].tolist()
stride_length_rf = gait_trimed['rf']['stride_length'].tolist()
stride_length_rr = gait_trimed['rr']['stride_length'].tolist()

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
stride_frequency_lf = gait_trimed['lf']['cadence'].tolist()
stride_frequency_lr = gait_trimed['lr']['cadence'].tolist()
stride_frequency_rf = gait_trimed['rf']['cadence'].tolist()
stride_frequency_rr = gait_trimed['rr']['cadence'].tolist()

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
ax[1].set_xlim([0,8])
ax[1].set_ylim([0,40])
ax[1].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[1].set_xlabel('stride frequency ($s^{-1}$)', fontsize=24, family='arial')
ax[1].set_ylabel('% of stride', fontsize=24, family='arial')
ax[1].spines[['top', 'right']].set_visible(False)
ax[1].spines['left'].set_linewidth(1)
ax[1].spines['bottom'].set_linewidth(1)

ax[2].set_box_aspect(1)
ax[2].set_xlim([0,400])
ax[2].set_ylim([0,30])
ax[2].tick_params(direction='out', width=1, length=10, labelsize=24)
ax[2].set_xlabel('stride speed (mm/s)', fontsize=24, family='arial')
ax[2].set_ylabel('% of stride', fontsize=24, family='arial')
ax[2].spines[['top', 'right']].set_visible(False)
ax[2].spines['left'].set_linewidth(1)
ax[2].spines['bottom'].set_linewidth(1)

plt.tight_layout()
plt.savefig(f'{figpath}Fig1-JKL.pdf',dpi=300,bbox_inches='tight',transparent=True)
#%% plot one gait cycle for illustration Figure1B

walk_id = 27
wk = stride_trimed['lr']['stride'][walk_id,:]

t = np.arange(wk[0], wk[2], 1/params['fr'])
index = np.arange(int(np.round(wk[0]*params['fr'])), \
                  int(np.round(wk[1]*params['fr'])), 1)
N = min(t.shape,index.shape)[0]
t = t[:N]-t[0]
index = index[:N]

ex_win = 0.05 #5min pre/post
wk_ex = [max(wk[0]-ex_win, 0), min(wk[2]+ex_win, params['rec_duration'])]
t_ex = np.arange(wk_ex[0], wk_ex[1], 1/params['fr'])
index_ex = np.arange(int(np.round(wk_ex[0]*params['fr'])), \
                  int(np.round(wk_ex[1]*params['fr'])), 1)
N = min(t_ex.shape,index_ex.shape)[0]
t_ex = t_ex[:N]-t_ex[0]
index_ex = index_ex[:N]

f, ax0 = plt.subplots(1, 1, figsize=(1.5, 0.5))

ax0.plot(t_ex, svm_bf['lr'][index_ex], color = '0', lw=0.5)
ax0.spines[['left','top', 'right', 'bottom']].set_visible(False)
ax0.axis('off')
#plt.savefig(f'{output}gait_cycle_{mouse_id[-1]}.pdf', dpi=300, \
#            bbox_inches='tight', transparent=True)
plt.savefig(f'{figpath}Fig1-B.pdf',dpi=300,bbox_inches='tight',transparent=True)

#%% plot limb position vs time & limb phase vs time Figure2D
#F2203-LR

walk_id = 27
wk = walkTimes[walk_id,:]
t = np.arange(wk[0], wk[1], 1/params['fr'])
index = np.arange(int(np.round(wk[0]*params['fr'])), \
                  int(np.round(wk[1]*params['fr'])), 1)
N = min(t.shape,index.shape)[0]
t = t[:N]-t[0]
index = index[:N]
# take 1s data
t = t[t.shape[0]-81:-30]
t = t-t[0]
index = index[index.shape[0]-81:-30]

f, ax0 = plt.subplots(1, 1, figsize=(3.5, 2))
ax0.plot(t, svm_bf['lr'][index], lw=1, color = '#000000')
ax0.set_ylim([-16, 16])
ax0.set_yticks([-16, -8, 0, 8, 16])
ax0.set_yticklabels(ax0.get_yticks(), family='arial')
ax0.tick_params(direction='out', width=0.5, length=4, labelsize=7)
ax0.set_xlim([t[0], t[-1]])
ax0.spines[['top', 'bottom']].set_visible(False)
ax0.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax0.set_ylabel('Limb position (mm)', fontsize=7, family='arial')
ax0.tick_params(direction='out', width=1, length=5, labelsize=7)
fontprops = fm.FontProperties(family='arial',size=7)
scalebar = AnchoredSizeBar(ax0.transData,
                           0.1, '0.1 s', 
                           loc = 'lower right', 
                           bbox_to_anchor=(0.5, -0.2),
                           bbox_transform=ax0.transAxes,
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1,
                           fontproperties=fontprops)

ax0.add_artist(scalebar)
# save data for graphpad prism - Figure 2D
position2time_fr = zip(t.tolist(), svm_bf['lr'][index].tolist())
position2time_fr_df = pd.DataFrame(position2time_fr, columns=['time', 'position'])

ax1 = ax0.twinx()
ax1.spines[['top', 'bottom']].set_visible(False)
ax1.spines['right'].set_color('red')
ax1.plot(t, svm_angle['lr'][index], lw=1, color = '#FF0000')
ax1.set_ylim([0, 360])
ax1.set_yticks([0, 360])
ax1.set_yticklabels(ax1.get_yticks(), family='arial')
ax1.tick_params(colors='red', direction='out', width=0.5, length=4, labelsize=7)
ax1.set_ylabel('limb phase (deg)',fontsize=7, family='arial')
ax1.yaxis.label.set_color('red')
#ax1.set_xlim([0, 1])
# save data for graphpad prism
phase2time_fr = zip(t.tolist(), svm_angle['lr'][index].tolist())
phase2time_fr_df = pd.DataFrame(phase2time_fr, columns=['time', 'phase'])
#phase2time_fr_df.to_csv(f'{output}phase2time_{mouse_id[-1]}_lr.csv')
plt.savefig(f'{figpath}Fig2-D.pdf', dpi=300, bbox_inches='tight', transparent=True)