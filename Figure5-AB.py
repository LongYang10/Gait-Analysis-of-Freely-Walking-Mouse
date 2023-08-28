# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:31:16 2023
@author: lonya
"""
#%% preset
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import messagebox
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import numpy as np
#%% Load Data
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Choose F1590.mat or F2275.mat',\
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
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
#%% processing data
print('load file: ' + all_files[0])
mouse_id = all_files[0][len(all_files[0])-9:-4]

if mouse_id=='F1590':
    strain='D1'
    group_id='CT'
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
# get locomotion trajectory
mouse2D = mf.mouse2D_format(eMouse)
walkTimes = mouse2D['walkTimes']
params = mouse2D['params']

# calculate head angle
nose_tail_v = mouse2D['nose']-mouse2D['tail']
params['minimum_turn'] = 20
head_angles, turn_labels = mf.walk_angle_change(nose_tail_v, \
                                                walkTimes,\
                                                params)
#%% figure setup
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42  

#%% plot the whole body trajectory - Figure5A,B
# 2 example animals
# sham lesion: F1590
# 6OHDA leiosn: F2275

fig = plt.figure(figsize=(3,6))
ax1 = fig.add_subplot(211)
bd = mouse2D['body']
walk_df = pd.DataFrame()
for i in range(int(walkTimes.shape[0])):
    ind1 = int(walkTimes[i,0]*params['fr'])
    ind2 = int(walkTimes[i,1]*params['fr'])
    color = range(ind2-ind1)
    ax1.scatter(bd[ind1:ind2,0], bd[ind1:ind2,1], s=0.1,c=color,cmap= 'Greys')
    temp_walk = pd.DataFrame(bd[ind1:ind2,:])
    walk_df = pd.concat([walk_df, temp_walk], ignore_index = True)
# save data for graphpad prism
#walk_df.columns = ['x', 'y']
#walk_df.to_csv(f'{output}walk_trajectory_{mouse_id[-1]}.csv')
    
ax1.set_xlim([0, 630])
ax1.set_ylim([0, 630])
ax1.set_xlabel('position x (mm)', fontsize=20, family='arial')
ax1.set_ylabel('position y (mm)', fontsize=20, family='arial')
ax1.tick_params(direction='out', width=2, length=5, labelsize=24)
ax1.set_xticks([0, 600])
ax1.set_xticklabels(ax1.get_xticks(), family='arial')
ax1.set_yticks([0, 600])
ax1.set_yticklabels(ax1.get_yticks(), family='arial')
ax1.set_aspect('equal', adjustable='box')
ax1.spines['left'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)
ax1.spines['right'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
#ax1.invert_yaxis()

ax2 = fig.add_subplot(212)
bouts = walkTimes
walk_angle = []
walk_mean_angle = np.zeros(bouts.shape[0])
for i in range(bouts.shape[0]):
    head_angle = []
    walk_bout = bouts[i,:]
    t = np.arange(walk_bout[0], walk_bout[1], 1/params['fr'])
    walk_id = np.arange(int(walk_bout[0]*params['fr']), \
                        int(walk_bout[1]*params['fr']),1)
    N = min(t.shape,walk_id.shape)[0]
    t, walk_id = t[:N-1], walk_id[:N-1]
    for j in range(walk_id.shape[0]):
        angle = mf.angle_2vectors(nose_tail_v[walk_id[0],:], \
                               nose_tail_v[walk_id[j],:]) # current frame angle
        head_angle = np.append(head_angle, angle) # save stride
    head_angle = np.rad2deg(head_angle)
    ax2.plot(t-t[0], head_angle, lw=0.5, color='darkgrey')
    walk_angle.append(head_angle)
    walk_mean_angle[i] = np.mean(head_angle)

ax2.set_xlabel('time (s)', fontsize=20, family='arial')
ax2.set_ylabel('walking direction (deg)', fontsize=20, family='arial')
ax2.set_xlim([0,3])
ax2.set_xticks([0, 1, 2, 3])
ax2.tick_params(width=0.5)
ax2.set_xticklabels(ax2.get_xticks(),fontsize=18, family='arial')
ax2.set_ylim([-180, 180])
ax2.set_yticks([-180, -90, 0, 90, 180])
ax2.set_yticklabels(ax2.get_yticks(), fontsize=18, family='arial')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(2)
ax2.spines['bottom'].set_linewidth(2)

plt.tight_layout()

if mouse_id=='F1590':
    plt.savefig(f'{figpath}Figure5-A.pdf', dpi=300,bbox_inches='tight', transparent=True)
elif mouse_id=='F2275':
    plt.savefig(f'{figpath}Figure5-B.pdf', dpi=300,bbox_inches='tight', transparent=True)
else:
    plt.savefig(f'{figpath}Figure5-{mouse_id}.pdf', dpi=300,bbox_inches='tight', transparent=True)