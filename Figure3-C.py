# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:11:11 2023

@author: lonya
"""
#%% Load Data
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn3

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

figpath = str(output_path) + 'Figure 3/'

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
    
#%% pie plot of limb phase locking, speed encoding, start/stop encoding

# phase locking encoding (at least one limb)
psi_p_hl_lf = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf'),['p']].values
psi_p_hl_lr = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lr'),['p']].values
psi_p_hl_rf = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='rf'),['p']].values
psi_p_hl_rr = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='rr'),['p']].values

psi_mtx = np.concatenate((psi_p_hl_lf,psi_p_hl_lr,psi_p_hl_rf,psi_p_hl_rr),axis=1)

psi_flag = np.zeros_like(psi_mtx)
psi_flag[np.where(psi_mtx<0.05)] = 1
psi_flag_sum = np.sum(psi_flag, axis=1)

psi_tag = np.zeros(psi_mtx.shape[0])
psi_tag[np.where(psi_flag_sum>=1)]=1

group_psi = np.where(psi_tag==1)[0]

# body speed encoding
all_pval = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf'),['speed_score_p']].values
group_speed = np.where(all_pval<0.05)[0]

# start/stop encoding
all_start_p = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf'),['start_score_p']].values
all_stop_p = psi_trimed.loc[(psi_trimed['group_id']=='HL')&\
                           (psi_trimed['limb']=='lf'),['stop_score_p']].values
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
'''
# for illustration purpose, increased the area size
out = venn3_unweighted(subsets = (5, 90, 48, 7, 5, 55, 51),
      set_labels = labels, alpha = 0.5,
      set_colors=('red','green','blue'),
      subset_areas = (5, 90, 48, 7, 5+10, 55, 51),
      subset_label_formatter=lambda x: f"{(x/total):1.0%}");
'''
for text in out.set_labels:
   text.set_fontsize(7)
for text in out.subset_labels:
    if text is not None:
        text.set_fontsize(7)  

plt.savefig(f'{figpath}Figure3-C.pdf',dpi=300,bbox_inches='tight', transparent=True)