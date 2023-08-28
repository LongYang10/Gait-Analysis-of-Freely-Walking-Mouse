# -*- coding: utf-8 -*-
"""
calculate encoding index of all animals (HL, CT, PD), and save the result

1. phase locking index
2. speed encoding score
3. start/stop encoding index

significance test: spike time jitter test
function: get_stride_index
jitter window: [-0.5, 0.5]
shuffling times: 100

@author: longyang
"""
#%% Load Data
from tkinter import messagebox
import tkinter as tk
import tkinter.filedialog as fd
import pandas as pd
import scipy.io as sio
import my_funcs as mf
import pickle
import os
#%% Select animals data
root = tk.Tk()
def main():
    files = fd.askopenfilenames(parent=root,\
                                title='Choose tagging eMouse.mat files',\
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
#%% Load previous saved data path
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, title='Choose: all-mice-datapath.pickle')
root.destroy()
print(filez)

with open(filez[0], 'rb') as f:
    all_files = pickle.load(f)
#%% processing data
mouse_id = []
group_id = []
strain = []
stage = []
neuron_index = pd.DataFrame()
for file in all_files: # loop through all selected files
    # find mouse related info (e.g. id, strain, group) from folder structure
    print('load file: ' + file)
    backslash_id = [m for m,x in enumerate(file) if x=='/']
    mouse_id.append(file[backslash_id[-3]+1:backslash_id[-2]])
    strain.append(file[backslash_id[-4]+1:backslash_id[-3]])
    group_id.append(file[backslash_id[-5]+1:backslash_id[-4]])
    stage.append(file[backslash_id[-2]+1:backslash_id[-1]])
    # load eMouse file
    matfile = sio.loadmat(file, squeeze_me = True)
    eMouse = matfile['eMouse']
    # get stimuli
    pulseTimes = eMouse['stimuli'].item()['pulseTimes'].item()
    laserOn = eMouse['stimuli'].item()['pulseTrainTimes'].item()
    # get 2D coordination (world-view) & manually curated walking bouts
    mouse2D = mf.mouse2D_format(eMouse)
    walkTimes = mouse2D['walkTimes']
    params = mouse2D['params']
    # get self-view mouse
    svm = mf.createSVM(mouse2D)
    # get filtered self-view mouse
    svm_bf = mf.mouse_filter(mouse = svm, fwin=[0.5,8], ftype='band')
    # get gaits & strides
    gait, stride = mf.calculate_gait(svm_bf, mouse2D)
    gait_trimed, stride_trimed = mf.trim_gait(gait, stride)
    # get kinematics
    kinematics = mf.mouse_kinematics(mouse2D, params)
    # prepare ephys data
    st_mtx, camTimes, tagging, cln, clwf, fs, xcoords, ycoords, unitBestCh = \
        mf.ephys_format(eMouse)
    # get opto-tagging data
    new_tagging, tagss = mf.get_tagging(st_mtx, clwf, fs, pulseTimes, params)
    # calculate single neuron phase locked to gait cycle
    params['n_circ_bins'] = 24
    params['density'] = False
    params['norm']=True
    ### calculate limb phase angle
    svm_angle = mf.get_mouse_angle_pct(svm_bf, stride_trimed)
    # calculate single neuron encoding index of PHASE, SPEED, START/STOP
    params['repeat']=100
    stride_index = mf.get_stride_index(svm_angle, kinematics,\
                                       st_mtx, stride_trimed, params)
    # save all data into dataframe
    index_df = mf.get_stride_df(stride_index, tagging, xcoords, ycoords, unitBestCh,\
                           mouse_id[-1], group_id[-1], strain[-1], \
                           stage[-1], kw = tagss[['latency','baseline_fr']])
    neuron_index = pd.concat([neuron_index, index_df], ignore_index = True)

#%% create path for figures
backslash_id = [i for i,x in enumerate(all_files[0]) if x=='/']
output_path = all_files[0][:backslash_id[-1]+1]
figpath = str(output_path) + 'Figure 5/'
if not os.path.exists(figpath):
    os.makedirs(figpath)

#%% Save data analysis results
print('data saved to: ' + output_path)

with open(output_path + 'all-mice-index.pickle', 'wb') as f:
    pickle.dump(neuron_index, f)
#Save gaits data into excel
neuron_index.to_csv(output_path + 'all-mice-index.csv')