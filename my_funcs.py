# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 13:14:34 2023
@author: lonya
"""
# # local function
# In[36]:
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from scipy.stats import sem

def mouse2D_format(eMouse):
    '''
    convert matlab data into python dict
    input: eMouse - 
    output: mouse2D - dict
    '''
    # walkTimes
    mouse2D = eMouse['mouse2D'].item()
    fr = mouse2D['params'].item()['fr'].item()
    camTimes = eMouse['stimuli'].item()['camTimes'].item()
    pulseTimes = eMouse['stimuli'].item()['pulseTimes'].item()
    
    confirmed_walk_frames = eMouse['confirmed_walk_frames'].item()
    startframes = confirmed_walk_frames['startframes'].item()
    stopframes = confirmed_walk_frames['stopframes'].item()

    walkTimes = np.concatenate((np.expand_dims(startframes, axis=1), \
                                np.expand_dims(stopframes, axis=1)),axis=1)/fr
    # fill the pre-camera on time with dumy frames, all 0
    dumy_frame = np.zeros((int(np.round(camTimes[0]*fr)),2))
    lf = np.concatenate((dumy_frame, mouse2D['lf'].item()))
    lr = np.concatenate((dumy_frame, mouse2D['lr'].item()))
    rf = np.concatenate((dumy_frame, mouse2D['rf'].item()))
    rr = np.concatenate((dumy_frame, mouse2D['rr'].item()))
    nose = np.concatenate((dumy_frame, mouse2D['nose'].item()))
    tail = np.concatenate((dumy_frame, mouse2D['tail'].item()))
    px2mm = mouse2D['params'].item()['px2mm'].item()
    
    body = (lf+lr+rf+rr+nose+tail)/6
    rec_duration = lf.shape[0]/fr #whole recording time aligned to ephys
    
    params = {'px2mm': px2mm, 'fr': fr, 'rec_duration': rec_duration, \
              'camTimes': camTimes}
    mouse2D = {}
    mouse2D = {'lf': lf, 'lr': lr, 'rf': rf, 'rr': rr, 'nose': nose, \
               'tail': tail,'params': params, 'body': body, \
               'walkTimes': walkTimes, 'pulseTimes': pulseTimes}
    # smooth mouse2D
    if not('smn' in params.keys()):
        params['smn'] = 11 #smooth window
    mouse2D_sm = smooth_mouse(mouse = mouse2D, win=params['smn'])
    #mouse2D_sm = mouse2D
    #1/11/2023, for debugs
    #
    return mouse2D_sm

def smooth_mouse(mouse, win, poly=3):
    """
    mouse is a dictionary, including 'lf', 'rf'
    mouse['lf'] is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    mouse_sm = {}
    lf = np.zeros_like(mouse['lf'])
    lr = np.zeros_like(mouse['lf'])
    rf = np.zeros_like(mouse['lf'])
    rr = np.zeros_like(mouse['lf'])
    nose = np.zeros_like(mouse['lf'])
    tail = np.zeros_like(mouse['lf'])
    for c in range(mouse['lf'].shape[-1]):
        lf[:, c] = savgol_filter(mouse['lf'][:, c], win, poly)
        rf[:, c] = savgol_filter(mouse['rf'][:, c], win, poly)
        lr[:, c] = savgol_filter(mouse['lr'][:, c], win, poly)
        rr[:, c] = savgol_filter(mouse['rr'][:, c], win, poly)
        nose[:, c] = savgol_filter(mouse['nose'][:, c], win, poly)
        tail[:, c] = savgol_filter(mouse['tail'][:, c], win, poly)

    mouse_sm['lf'] = lf
    mouse_sm['rf'] = rf
    mouse_sm['lr'] = lr
    mouse_sm['rr'] = rr
    mouse_sm['nose'] = nose
    mouse_sm['tail'] = tail
    
    # get the body-centroid coordination
    body = (mouse_sm['lf'] + mouse_sm['rf'] + mouse_sm['lr'] + \
            mouse_sm['rr'] + mouse_sm['tail'] + mouse_sm['nose']) / 6
    mouse_sm['body'] = body
    mouse_sm['params'] = mouse['params']
    mouse_sm['walkTimes'] = mouse['walkTimes']
    mouse_sm['pulseTimes'] = mouse['pulseTimes']
    #mouse_sm['camTimes'] = mouse['camTimes']
    return mouse_sm

def ephys_format(eMouse):
    '''
    convert matlab data into python dict
    input: eMouse - 
    output: mouse2D - dict
    '''
    stimuli = eMouse['stimuli'].item()
    camTimes = stimuli['camTimes'].item()
    ephys = eMouse['ephys'].item()

    st = ephys['spikeTimes'].item()
    cl = ephys['spikeCluster'].item()
    cln = ephys['clusterName'].item()
    #clid = ephys['clusterID'].item()
    clwf = ephys['clusterWaveform'].item()
    fs = ephys['fs'].item()
    if 'tagging' in ephys.dtype.names:
        tagging = ephys['tagging'].item()
    else:
        tagging = np.zeros(len(cln))
    # add probe geometry
    if 'xcoords' in ephys.dtype.names:
        xcoords = ephys['xcoords'].item()
        ycoords = ephys['ycoords'].item()
        unitBestCh = ephys['clusterBestChannel'].item()
    else:
        xcoords = np.zeros(len(cln))
        ycoords = np.zeros(len(cln))
        unitBestCh = np.ones(len(cln)).astype(int)
    
    clu = np.unique(cl) # unique neurons
    nn = len(clu)       # neuron number
    # spike time matrix, row: neuron; column: spike times
    st_mtx = [] 
    for i in range(nn):
        st_mtx.append(st[np.where(cl == clu[i])[0]])
    st_mtx = np.array(st_mtx, dtype=object)
    
    return st_mtx, camTimes, tagging, cln, \
        clwf, fs, xcoords, ycoords, unitBestCh

def createSVM(mouse2D):
    '''
    create the self view mouse
    '''
    svm = {}
    camTimes = mouse2D['params']['camTimes']
    fr = mouse2D['params']['fr']
    ind = int(np.round(camTimes[0]*fr)) #end frame of dumy frames
    # generate 'tail' -> 'nose' vector
    nose_tail = mouse2D['tail'] - mouse2D['nose']
    nose_tail_mod = np.sqrt(np.square(nose_tail[:,0]) + np.square(nose_tail[:,1]))
    
    # generate 'left_forepaw' -> 'nose' vector
    nose_lf = mouse2D['lf'] - mouse2D['nose']
    # project "lf->nose" onto "tail->nose"
    nose_lf_mod = np.divide(np.multiply(nose_lf[ind:,0], nose_tail[ind:,0]) + \
                            np.multiply(nose_lf[ind:,1], nose_tail[ind:,1]), \
                                nose_tail_mod[ind:])
    # project "rf->nose" onto "tail->nose"
    nose_rf = mouse2D['rf'] - mouse2D['nose']
    nose_rf_mod = np.divide(np.multiply(nose_rf[ind:,0], nose_tail[ind:,0]) + \
                            np.multiply(nose_rf[ind:,1], nose_tail[ind:,1]), \
                                nose_tail_mod[ind:])
    # project "rr->nose" onto "tail->nose"
    nose_rr = mouse2D['rr'] - mouse2D['nose']
    nose_rr_mod = np.divide(np.multiply(nose_rr[ind:,0], nose_tail[ind:,0]) + \
                            np.multiply(nose_rr[ind:,1], nose_tail[ind:,1]), \
                                nose_tail_mod[ind:])
    #project "lr-nose" onto "tail->nose"
    nose_lr = mouse2D['lr'] - mouse2D['nose']
    nose_lr_mod = np.divide(np.multiply(nose_lr[ind:,0], nose_tail[ind:,0]) + \
                            np.multiply(nose_lr[ind:,1], nose_tail[ind:,1]), \
                                nose_tail_mod[ind:])
    
    #1/5/2023, calculate the distance between lf-rf and lr-rr to quantify the 
    #width variation
    lf_rf_v = mouse2D['lf']-mouse2D['rf']
    lf_rf = np.sqrt(lf_rf_v[:,0]**2+lf_rf_v[:,1]**2)
    lr_rr_v = mouse2D['lr']-mouse2D['rr']
    lr_rr = np.sqrt(lr_rr_v[:,0]**2+lr_rr_v[:,1]**2)
    svm['lf_rf'] = lf_rf
    svm['lr_rr'] = lr_rr

    svm['lf'] = np.concatenate((np.zeros(ind), nose_lf_mod))
    svm['rf'] = np.concatenate((np.zeros(ind), nose_rf_mod))
    svm['lr'] = np.concatenate((np.zeros(ind), nose_lr_mod))
    svm['rr'] = np.concatenate((np.zeros(ind), nose_rr_mod))
    svm['tail'] = nose_tail_mod
    svm['nose'] = mouse2D['nose']
    svm['params'] = mouse2D['params']
    svm['walkTimes'] = mouse2D['walkTimes']
    
    return svm

def mouse_filter(mouse,fwin,ftype):
    
    params = mouse['params']
    filtered_Mouse = {}
    if ftype in 'band':
        sos = signal.butter(3, [fwin[0],fwin[1]], 'bandpass', \
                            fs=params['fr'], output='sos')
    if ftype in 'low':
        sos = signal.butter(3, fwin, 'lowpass', fs=params['fr'], output='sos')
    if ftype in 'high':
        sos = signal.butter(3, fwin, 'highpass', fs=params['fr'], output='sos')
    
    filtered_Mouse['lf'] = signal.sosfiltfilt(sos, mouse['lf'])
    filtered_Mouse['rf'] = signal.sosfiltfilt(sos, mouse['rf'])
    filtered_Mouse['lr'] = signal.sosfiltfilt(sos, mouse['lr'])
    filtered_Mouse['rr'] = signal.sosfiltfilt(sos, mouse['rr'])
    
    filtered_Mouse['lf_rf'] = signal.sosfiltfilt(sos, mouse['lf_rf'])
    filtered_Mouse['lr_rr'] = signal.sosfiltfilt(sos, mouse['lr_rr'])
    
    filtered_Mouse['tail'] = mouse['tail']
    
    params['mouse_filter_window'] = fwin
    params['mouse_filter_type'] = ftype
    filtered_Mouse['params'] = params
    filtered_Mouse['walkTimes'] = mouse['walkTimes'] #1/11/2023
    
    return filtered_Mouse

def get_mouse_angle_pct(mouse, stride):
    '''use percentage of stride as phase angle, set all the other phase angle
    as zero
    Args:
        mouse - (dictionary) vector of 4 limb trajectory, projected trajectory
            'lf','lr','rf','rr'
        stride
    Returns:
    dict (dictionary) with 2 named entries:
        stride - (numpy.ndarray) 3 column vector, stance_swing_stance phase times
        stride_flag - (numpy.ndarray) mark each stride with a walking bout flag
    '''
    lf = np.zeros(len(mouse['lf']))
    lr = np.zeros(len(mouse['lr']))
    rf = np.zeros(len(mouse['rf']))
    rr = np.zeros(len(mouse['rr']))
    
    lf_rf_a = hilbert(mouse['lf_rf'])
    lr_rr_a = hilbert(mouse['lr_rr'])
    lf_rf_angle = np.angle(lf_rf_a, deg=True)
    lr_rr_angle = np.angle(lr_rr_a, deg=True)
    lf_rf_angle = lf_rf_angle+180
    lr_rr_angle = lr_rr_angle+180
    
    params = mouse['params']
    fr = params['fr']

    for i in range(stride['lf']['stride'].shape[0]):
        ind1 = int(np.round(stride['lf']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['lf']['stride'][i,2]*fr))
        temp = np.linspace(0, 360, num=ind2-ind1+1, endpoint=True)
        lf[ind1:ind2+1] = temp
        
    for i in range(stride['lr']['stride'].shape[0]):
        ind1 = int(np.round(stride['lr']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['lr']['stride'][i,2]*fr))
        lr[ind1:ind2+1] = np.linspace(0, 360, num=ind2-ind1+1, endpoint=True)
        
    for i in range(stride['rf']['stride'].shape[0]):
        ind1 = int(np.round(stride['rf']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['rf']['stride'][i,2]*fr))
        rf[ind1:ind2+1] = np.linspace(0, 360, num=ind2-ind1+1, endpoint=True)
        
    for i in range(stride['rr']['stride'].shape[0]):
        ind1 = int(np.round(stride['rr']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['rr']['stride'][i,2]*fr))
        rr[ind1:ind2+1] = np.linspace(0, 360, num=ind2-ind1+1, endpoint=True)
    
    return {'lf':lf, 'lr':lr, 'rf':rf, 'rr':rr, 'params': params,\
            'lf_rf': lf_rf_angle, 'lr_rr': lr_rr_angle}

def get_stride_angle(mouse_angle, stride):
    """select the phase angles during stride, 1/9/2023"""
    params = mouse_angle['params']
    fr = params['fr']

    mouse_stride_angle = {}
    lf, lr, rf, rr = np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(stride['lf']['stride'].shape[0]):
        ind1 = int(np.round(stride['lf']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['lf']['stride'][i,2]*fr)) 
        lf = np.concatenate((lf,mouse_angle['lf'][ind1:ind2]),axis=None)
    for i in range(stride['lr']['stride'].shape[0]):
        ind1 = int(np.round(stride['lr']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['lr']['stride'][i,2]*fr))
        lr = np.concatenate((lr,mouse_angle['lr'][ind1:ind2]),axis=None)
    for i in range(stride['rf']['stride'].shape[0]):
        ind1 = int(np.round(stride['rf']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['rf']['stride'][i,2]*fr))
        rf = np.concatenate((rf,mouse_angle['rf'][ind1:ind2]),axis=None)
    for i in range(stride['rr']['stride'].shape[0]):
        ind1 = int(np.round(stride['rr']['stride'][i,0]*fr))
        ind2 = int(np.round(stride['rr']['stride'][i,2]*fr))
        rr = np.concatenate((rr,mouse_angle['rr'][ind1:ind2]),axis=None)
        
    mouse_stride_angle['lf'] = lf
    mouse_stride_angle['lr'] = lr
    mouse_stride_angle['rf'] = rf
    mouse_stride_angle['rr'] = rr
    mouse_stride_angle['params'] = params
    
    return mouse_stride_angle

def get_stride_hist(stride_angle, params):
    '''Calculate the angular histogram of phase angle during stride
    '''
    if not('n_circ_bins' in params.keys()):
        params['n_circ_bins'] = 24        # number of bins for angular histogram, default=24
    nb = params['n_circ_bins']
    lf, bin_edges = np.histogram(stride_angle['lf'], bins = nb, range=(0, 360), density=params['density'])
    lr, _ = np.histogram(stride_angle['lr'], bins = nb, range=(0, 360), density=params['density'])
    rf, _ = np.histogram(stride_angle['rf'], bins = nb, range=(0, 360), density=params['density'])
    rr, _ = np.histogram(stride_angle['rr'], bins = nb, range=(0, 360), density=params['density'])
    hist = {'lf': lf, 'lr': lr, 'rf': rf, 'rr': rr}
    
    return hist, bin_edges

def t2idx(t, sr, N):
    '''turn time period into index.
    input:  t -[mXn], m time windows; 1st column is the start of time, 
                n-th column is the end of time.
            sr - sampling rate.
    output: idx - array of 0/1, 1 show when the event happened'''
    idx = np.zeros(N)
    for i in range(t.shape[0]):
        ind1, ind2 = int(np.round(t[i,0]*sr)), int(np.round(t[i,t.shape[1]-1]*sr))
        n = np.arange(ind1,ind2) #3/29/2023, index shift to right
        idx[n] = 1
    return idx

def get_spike_phase_angle(angle, st_mtx, sub_flag, fr):
    '''pick out the phase angles when specific event happened, like neuron firing, walking
    input - angle, phase angle.
            st_mtx, spike times tuple, m rows, m is the number of neurons; each row may have 
            different length, depending on how many spikes detected.
            sub_flag, 2nd event, if you have other conditions
            fr, frame rate
    output - spa_mtx, spike phase angle tuple, m rows, same as st_mtx; each row have the phase of 
            spikes selected by sub_flag
    '''
    spa_mtx=[]
    #pdb.set_trace()
    for i in range(st_mtx.shape[0]):
        sp_flag = np.zeros(len(angle))
        st = st_mtx[i]
        st = st[np.where((st<(len(sub_flag)-1)/fr) & (st>0))[0]]
        sp_idx = np.round(st*fr).astype(int)
        sp_flag[sp_idx] = 1
        # find the intersection between two events
        idx = np.where((sub_flag+sp_flag)==2)[0]
        spa_mtx.append(angle[idx])
    
    return np.array(spa_mtx, dtype=object)

def get4limb_spa_stride(angle, st_mtx, stride):
    '''get 4 limbs spike phase angle during strides
    input - angle (dictionary), including 'lf'/'lr'/'rf'/'rr' 4limbs phase angles.
            st_mtx (tuple), all neurons spike times, m row equal to number of neurons
            stride (dictionary), including 4limbs stride times
    output - flimb_spa (dictionary), including 4limbs spike phase angles during stride
    '''
    fr = angle['params']['fr']
    flimb_spa={}
    N = len(angle['lf']) # the length of total phase angles
    
    # calculate the stride flags of each limb
    lf_str_flag = t2idx(stride['lf']['stride'], fr, N)
    lr_str_flag = t2idx(stride['lr']['stride'], fr, N)
    rf_str_flag = t2idx(stride['rf']['stride'], fr, N)
    rr_str_flag = t2idx(stride['rr']['stride'], fr, N)
    
    # calculate the spike phase angles of each limb
    flimb_spa['lf'] = get_spike_phase_angle(angle['lf'], st_mtx, lf_str_flag, fr)
    flimb_spa['lr'] = get_spike_phase_angle(angle['lr'], st_mtx, lr_str_flag, fr)
    flimb_spa['rf'] = get_spike_phase_angle(angle['rf'], st_mtx, rf_str_flag, fr)
    flimb_spa['rr'] = get_spike_phase_angle(angle['rr'], st_mtx, rr_str_flag, fr)
    
    return flimb_spa

def get_spa_hist(spa, base, nb, fr, norm,dens):
    '''calculate the spike phase angles distribution. (optional: normalized by the continuous
    phase angles distribution)
    input - spa (tuple), spike phase angles, m rows equal to number of neurons
            base (list), continuous phase angles distribution
            nb, number of bins
            fr, frame rate
            norm (bool), optional normalization
    output - spa_d
    '''
    nn = spa.shape[0] # number of neurons
    spa_d = np.empty((nn,nb))
    for i in range(nn):
        spa_d[i,:], bin_edges = np.histogram(spa[i], bins = nb, range=(0, 360), density=dens)
    if norm:
        base_mtx = np.repeat(np.expand_dims(base, axis=0), nn, axis=0) # repeat nn times
        spa_d = np.divide(spa_d, base_mtx)*fr #normalize spike phase angle distribution, 
                                            #turn into firing rate
    return spa_d

def get4limb_spah_stride(spa, base, params):
    '''calculate 4limbs spike phase angles distribution during stride. 
        (optional: normalized by the continuous phase angles distribution)
    input - spa (dict), 4limbs spike phase angles during stride, m rows equal to number of neurons
            base (dict), 4limbs continuous phase angles distribution druing stride
            params (dict) - 'n_circ_bins' number of bins for histogram;
                            'fr': frame rate
    output - spad
    '''
    if not('n_circ_bins' in params.keys()):
        params['n_circ_bins'] = 24 # default 15 degree bin size
    nb, fr, norm, dens = params['n_circ_bins'], params['fr'], params['norm'],params['density']
    flimb_spad = {}
    flimb_spad['lf'] = get_spa_hist(spa['lf'], base['lf'], nb, fr, norm,dens)
    flimb_spad['lr'] = get_spa_hist(spa['lr'], base['lr'], nb, fr, norm,dens)
    flimb_spad['rf'] = get_spa_hist(spa['rf'], base['rf'], nb, fr, norm,dens)
    flimb_spad['rr'] = get_spa_hist(spa['rr'], base['rr'], nb, fr, norm,dens)
    
    return flimb_spad

def circ_r_group(angles, base, params):
    '''
    Circular Analysis
    ref: 1. Zar, J. H. Biostatistical Analysis. (2014). p653
    2/2/2022 - group data'''
    if len(angles):
        nb = params['n_circ_bins']
        fr = params['fr']
        #angles = np.deg2rad(angles) # added 12/9/2022, after save angles in degree
        hist, bin_edges = np.histogram(angles, nb, range=(0, 360),density=params['density'])
        #angles_group = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
        angles_group = bin_edges[:-1] #1/17/2023, 
        if params['norm']:
            hist = np.divide(hist,base)*fr
        #hist_sm = gaussian_filter(hist, sigma = 3, mode='wrap')
        
        hist_sm = hist#1/13/2023
        #n = angles.shape[0]
        #12/13/2022, change 
        n = np.sum(hist_sm)

        Y = np.sum(hist_sm*np.sin(angles_group*np.pi/180))/n
        X = np.sum(hist_sm*np.cos(angles_group*np.pi/180))/n

        r = np.sqrt(Y**2 + X**2)                    # vector length
        a = np.arctan2(Y,X)                  # mean angle (in radians) 1/9/2023,+pi
        #if a<0:
        #    a = a+2*np.pi
        #a = a+np.pi # keep it consistant with limb angle calculation
        ss = 2*(1-r)                                # angular variance

        # Rayleigh test
        R = n*r
        z = R**2/n
        p = np.exp(np.sqrt(1+4*n+4*(n**2-R**2)) - (1+2*n))      # p value for uniformity

        circ = np.array([r, a, p, np.log(z), ss])
    else:
        circ = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    return circ

def circ_mm(r, a):
    '''
    Circular Analysis - calculate the mean of mean angles of each neuron
    ref: 1. Zar, J. H. Biostatistical Analysis. (2014). p653
    input - r (numpy array), mean vector length of each neuron
            a (numpy array), mean vector angle of each neuron
    output - mr, mean vector length
             ma, mean vector angle
    '''
    r = r[~np.isnan(r)]
    a = a[~np.isnan(a)]
    k = len(r) #number of neurons
    X_mean = np.sum(r * np.cos(a * np.pi / 180.))/k
    Y_mean = np.sum(r * np.sin(a * np.pi / 180.))/k
    mr = np.sqrt(X_mean**2 + Y_mean**2)
    ma = np.arctan2(Y_mean,X_mean)
    if ma<0:
        ma = ma + 2*np.pi
    ma = ma*180/np.pi
    
    return mr, ma

def circ_m(a):
    a = a[~np.isnan(a)]
    k = len(a) #number of neurons
    X = np.sum(np.cos(a * np.pi / 180.))/k
    Y = np.sum(np.sin(a * np.pi / 180.))/k
    r = np.sqrt(X**2 + Y**2)
    
    ma = np.arctan2(Y,X)
    if ma<0:
        ma = ma+2*np.pi
    ma = ma*180/np.pi
        
    #s0 = 180/np.pi*np.sqrt(-1*4.60517*np.log(r))
    s0 = 180/np.pi*np.sqrt(2*(1-r))
    return ma, s0, r

def permut_angle(a,b,n):
    a_m,_,a_r = circ_m(a)
    b_m,_,b_r = circ_m(b)
    
    #d_a_b = 2-2*np.cos(np.deg2rad(a_m-b_m)) # unit vector length
    d_a_b = np.sqrt(a_r**2 +b_r**2-2*a_r*b_r*np.cos(np.deg2rad(a_m-b_m)))
    
    mix = np.concatenate((a,b))
    d_mixed = np.zeros((n,1))
    
    for i in range(n):
        rng = np.random.default_rng()
        new_index = rng.permutation(mix.shape[0])
        
        A = mix[new_index[:a.shape[0]],:]
        B = mix[new_index[a.shape[0]:],:]
        
        A_m,_,A_r = circ_m(A)
        B_m,_,B_r = circ_m(B)

        #d_mixed[i,0] = 2-2*np.cos(np.deg2rad(A_m-B_m))
        d_mixed[i,0] = np.sqrt(A_r**2 +B_r**2-2*A_r*B_r*np.cos(np.deg2rad(A_m-B_m)))
    return len(np.where(d_mixed>d_a_b)[0])/n

def get4limb_spa_circ_stride(spa, base, params):
    '''get 4limbs spike phase angles circular analysis
    input - spa (dict),  4limbs spike phase angles during stride
            base (dict), 4limbs continuouse phase angles distribution during stride
            params (dict), 'norm' normalization
    output - flimb_spac'''
    flimbs_spac = {}
    nn = spa['lf'].shape[0]
    lf,lr,rr,rf = np.zeros([nn,5]),np.zeros([nn,5]),np.zeros([nn,5]),np.zeros([nn,5])
    for i in range(nn):
        #pdb.set_trace()
        lf[i,:] = circ_r_group(spa['lf'][i], base['lf'], params)
        lr[i,:] = circ_r_group(spa['lr'][i], base['lr'], params)
        rf[i,:] = circ_r_group(spa['rf'][i], base['rf'], params)
        rr[i,:] = circ_r_group(spa['rr'][i], base['rr'], params)
    
    flimbs_spac={'lf':lf, 'lr':lr, 'rf':rf, 'rr':rr}
    return flimbs_spac

def shuffle_spike_times_full(st_mtx, params):
    '''add different random time to each single spike'''
    min_shift = params['minimum_shift']
    max_shift = params['maximum_shift']
    #window = params['window']
    st = []
    for i in range(st_mtx.shape[0]):
        #generate random shift for each neuron
        shift_time = np.random.default_rng().uniform(min_shift,max_shift,len(st_mtx[i]))
        temp = st_mtx[i]+shift_time
        
        # tail wrapped to the head
        #temp[np.where(temp>max(temp))[0]] = \
        #    temp[np.where(temp>max(temp))[0]] - max(temp)
        #temp = np.sort(temp)
        st.append(temp)
        
    st = np.array(st, dtype=object)
    return st

def stride_time(tr, walkTimes, params): 
    '''systematically quantify the gaits, go through each detected walking period, find out stride that includes
    complete swing phase and stance phase.
    (8/1/2022: removed the swing_stance_swing; only counting stance_swing stride)
    Args:
        tr - (numpy.ndarray) vector of single limb trajectory, projected trajectory
    Returns:
    dict (dictionary) with 2 named entries:
        stride - (numpy.ndarray) 3 column vector, stance_swing_stance phase times
        stride_flag - (numpy.ndarray) mark each stride with a walking bout flag
    '''
    if not('peak_height' in params.keys()):
        params['peak_height'] = 2#4
    if not('peak_distance' in params.keys()):
        params['peak_distance'] = 3 #0.15
    if not('peak_width' in params.keys()):
        params['peak_width'] = 0.05
    if not('peak_prominence' in params.keys()):
        params['peak_prominence'] = 4
    
    rec_duration = params['rec_duration']
    stride = np.empty([0,3])
    stride_flag = []
    for i in range(walkTimes.shape[0]):
        walk_bout = walkTimes[i,:]
        if walk_bout[1] < rec_duration: #add this threshold for some animals have frame number > 144000
            t = np.arange(walk_bout[0], walk_bout[1], 1/params['fr'])
            walk_id = np.arange(int(np.round(walk_bout[0]*params['fr'])), \
                                int(np.round(walk_bout[1]*params['fr'])),1)
            #N = min(t.shape,walk_id.shape)[0]
            #t, walk_id = t[:N-1], walk_id[:N-1]
            
            '''pks,_ = signal.find_peaks(tr[walk_id], distance=params['peak_distance']*params['fr'],\
                prominence=params['peak_prominence'],width=params['peak_width'])
            vlys,_ = signal.find_peaks(-tr[walk_id], distance=params['peak_distance']*params['fr'],\
                                       prominence=params['peak_prominence'],\
                                       width=params['peak_width'])'''
            ###1/11/2023
            pks,pks_v = signal.find_peaks(tr[walk_id], \
                                      height=params['peak_height'],\
                                      distance=params['peak_distance'])
            vlys,vlys_v = signal.find_peaks(-tr[walk_id], \
                                       height=params['peak_height'],\
                                       distance=params['peak_distance'])

            if pks.shape[0]>1 and vlys.shape[0]>1:
                pks_t, vlys_t = t[pks], t[vlys]
                if pks[0] < vlys[0]: # first point is swing
                    for j in range(pks.shape[0]-1): # loop search all stride cycle
                        if j<vlys.shape[0]-1:
                            if vlys[j]<pks[j+1] and vlys[j+1]>pks[j+1]: # detect stride cycle
                                st_sw_st = np.array([vlys_t[j], pks_t[j+1], vlys_t[j+1]], ndmin=2)
                                stride = np.append(stride, st_sw_st, axis=0) # save stride
                                stride_flag = np.append(stride_flag, i)
                else: # first point is stance
                    for j in range(pks.shape[0]): # loop search all stride cycle
                        if j<vlys.shape[0]-1: 
                            if pks[j]>vlys[j] and pks[j]<vlys[j+1]: # detect stride cycle
                                st_sw_st = np.array([vlys_t[j], pks_t[j], vlys_t[j+1]], ndmin=2)
                                stride = np.append(stride, st_sw_st, axis=0) # save stride   
                                stride_flag = np.append(stride_flag, i)
    if stride.shape[0]==0: 
        print('Warning!!! No stride cycle is detected!!!')
    return {'stride':stride, 'stride_flag':stride_flag}

def gaits_single_limb(tr, strideTimes, params):
    '''calculate the gaits of selected single limb
    Args:
    tr - (numpy.ndarray) vector of single limb trajectory, 2D mouse trajecotry
    strideTimes - dict (dictionary) with 2 named entries:
        stride - (numpy.ndarray) 3 column vector, stance_swing_stance phase times
        stride_flag - (numpy.ndarray) 
    Returns:
    dict (dictionary) with 7 named entries:
    speed - 
    stride_length - 
    cadence - 
    swing_stance_ratio - 
    swing_duration - 
    stance_duration - 
    swing_length - 
    '''
    stride_velocity, stride_length, cadence, swing_stance_ratio = [],[],[],[]
    swing_duration, stance_duration, swing_length = [], [], []
    fr = params['fr']
    sts= strideTimes['stride']
    
    if sts.shape[0]!=0:
        for i in range(sts.shape[0]):
            ind1, ind2 = int(np.round(sts[i,0]*fr)), int(np.round(sts[i,2]*fr))
            v_stride = tr[ind2,:] - tr[ind1,:]
            stride_length = np.append(stride_length, \
                                      np.sqrt(np.square(v_stride[0]) + \
                                              np.square(v_stride[1])))
            cadence = np.append(cadence, 1/(sts[i,2] - sts[i,0]))
            stance_duration = np.append(stance_duration, sts[i,1] - sts[i,0])
            swing_duration = np.append(swing_duration, sts[i,2] - sts[i,1])
            ind3 = int(np.round(sts[i,1]*fr))
            v_swing = tr[ind2,:] - tr[ind3,:]
            swing_length = np.append(swing_length, \
                                     np.sqrt(np.square(v_swing[0]) + \
                                             np.square(v_swing[1])))
    swing_stance_ratio = swing_duration/stance_duration
    stride_velocity = stride_length/(swing_duration + stance_duration)
    
    return {'stride_velocity': stride_velocity.tolist(),\
            'stride_length': stride_length.tolist(),\
            'cadence':cadence.tolist(), \
            'swing_stance_ratio': swing_stance_ratio.tolist(),\
            'swing_duration':swing_duration.tolist(),\
            'stance_duration': stance_duration.tolist(),\
            'swing_length': swing_length.tolist()}

def calculate_gait(mouse, mouse2D):

    # find the paired swing/stance time of each limb
    stride_lf = stride_time(mouse['lf'], mouse['walkTimes'], mouse['params'])
    stride_lr = stride_time(mouse['lr'], mouse['walkTimes'], mouse['params'])
    stride_rf = stride_time(mouse['rf'], mouse['walkTimes'], mouse['params'])
    stride_rr = stride_time(mouse['rr'], mouse['walkTimes'], mouse['params'])

    # calculate gaits of each limb
    gaits_lf = gaits_single_limb(mouse2D['lf'], stride_lf, mouse['params'])
    gaits_lr = gaits_single_limb(mouse2D['lr'], stride_lr, mouse['params'])
    gaits_rf = gaits_single_limb(mouse2D['rf'], stride_rf, mouse['params'])
    gaits_rr = gaits_single_limb(mouse2D['rr'], stride_rr, mouse['params'])

    stride = {'lf': stride_lf, 'lr': stride_lr, 'rf': stride_rf, 'rr': stride_rr}
    gait = {'lf': gaits_lf, 'lr': gaits_lr, 'rf': gaits_rf, 'rr': gaits_rr}
    
    return gait, stride

def trim_single_limb(gait, st):
    #1/11/2023
    gait = pd.DataFrame(data=gait)
    stride,stride_flag = st['stride'], st['stride_flag']
    good_bool = ((gait['stride_length']<=100) & \
                 (gait['stride_length']>5) & \
                 (gait['stride_velocity']<400) & \
                 (gait['stride_velocity']>0) & \
                 ((gait['swing_duration']+gait['stance_duration'])<1.5) & \
                 ((gait['swing_duration']+gait['stance_duration'])>0))
                #(gait['swing_stance_ratio']<2) & \
                #(gait['swing_stance_ratio']>0.1) & \
                #(gait['swing_length']<gait['stride_length'])) #80,5,450

    gait_trimed = gait.loc[good_bool]
    gait_trimed.reset_index(drop=True, inplace=True) #reset the index
    stride_trimed = stride[gait[good_bool].index,:]
    stride_flag_trimed = stride_flag[gait[good_bool].index]

    gait_trimed.to_dict('list')
    return gait_trimed, {'stride': stride_trimed, 'stride_flag': stride_flag_trimed}

def trim_gait(gait, stride):
    gait_trimed = {}
    stride_trimed = {}
    lfg, lfs = gait['lf'], stride['lf']
    lrg, lrs = gait['lr'], stride['lr']
    rfg, rfs = gait['rf'], stride['rf']
    rrg, rrs = gait['rr'], stride['rr']
    
    gait_trimed['lf'], stride_trimed['lf'] = trim_single_limb(lfg, lfs)
    gait_trimed['rf'], stride_trimed['rf'] = trim_single_limb(rfg, rfs)
    gait_trimed['lr'], stride_trimed['lr'] = trim_single_limb(lrg, lrs)
    gait_trimed['rr'], stride_trimed['rr'] = trim_single_limb(rrg, rrs)
    
    return gait_trimed, stride_trimed

def get_stride_index(angle, kinematics, st_mtx, stride, params):
    '''
    calculate the phase selectivity index of each neuron
    input
        angle - (dict)
            'lf': (numpy.ndarray) phase angle of left front limb trajectory
            'lr': same of left rear limb
            'rf': same of right front limb
            'rr': same of right rear limb
            'params': related parameters, like frame rate
        kinematics - (dict)
            'bd_speed': (numpy.ndarray) body speed
            'bd_speedUpTimes': (numpy.ndarray) start/stop of walking bouts
        st_mtx - (numpy.ndarray) row: neurons, column: spike times, so each row has different length
        stride - (numpy.ndarray) row: strides, column1: stance time, col2: swing time, col3: following stance time
    output
        psi - (dict)
            'lf': (numpy.ndarray) circular analysis results of each neurons, [n X 4] 
            n: number of neurons; column: [r, a, p, np.log(z)]
            'lr', 'rf', 'rr': same of each limb
            'lf_shuffle': (numpy.ndarray) r of each shuffled neurons, [n X 100]
            n: number of neurons; column: 100 repeat times
            'lr_shuffle', 'rf_shuffle', 'rr_shuffle': same of each limb
            'params': related parameters used in the funcion
    '''
    # PART1: phase locking index
    if not('n_circ_bins' in params.keys()):
        params['n_circ_bins'] = 24 # default phase bin size 15 degree
    if not('density' in params.keys()):
        params['density'] = False
    if not('norm' in params.keys()):
        params['norm'] = True
    ##1 calculate phase distribution of gait cycle
    stride_angle = get_stride_angle(angle, stride)
    stride_h, _ = get_stride_hist(stride_angle, params)
    ##2 pick out the phase angle when neuron firing during stride
    spa_str = get4limb_spa_stride(angle, st_mtx, stride)
    ##3 circular analysis
    psi = get4limb_spa_circ_stride(spa_str, stride_h, params)
    
    # PART 2: speed encoding index
    if not('speed_space' in params.keys()):
        params['speed_space'] = 10           # default speed bin size 10mm/s
    if not('speed_bin_valid_th' in params.keys()):
        params['speed_bin_valid_th'] = 0.005 # default at least 0.5% data included
    if not('maximum_speed' in params.keys()): 
        params['maximum_speed'] = 310        # default 300mm/s
    if not('remove_bad_speed_bins' in params.keys()):
        params['remove_bad_bins'] = False    #
    if not('rec_duration' in params.keys()):
        params['rec_duration'] = 1800        # default 30min
    if not('window' in params.keys()):
        params['window'] = [0, params['rec_duration']]
    ##1 binning speed & spikes
    binning_speed, binning_fr, _ = \
        calculate_speed_score(kinematics['bd_speed'],st_mtx, params)
    rval, pval = np.zeros(binning_fr.shape[1]), np.zeros(binning_fr.shape[1])
    ##2 calculate the pearson correlation coefficient between speed & spikes 
    for i in range(binning_fr.shape[1]):
        rval[i],pval[i] = pearsonr(binning_speed, binning_fr[:,i])
        
    # PART 3: start/stop encoding index
    _, _, _, _,start_p = psth_norm(kinematics['bd_start_stop'][:,0], \
                                   st_mtx, np.ones(len(st_mtx)), params) # start
    _, _, _, _,stop_p = psth_norm(kinematics['bd_start_stop'][:,1], \
                                  st_mtx, np.ones(len(st_mtx)), params) #stop
    
    '''neuron with larger r shows higher selectivity of phase angle 
    Criteria for significant phase locking: 
    vector length should be larger than 95 percentile of that of shuffled data'''
    # calculate the vector length of shuffled data
    # spikes of a given neuron were shifted by a random time between -0.5s to 0.5s
    # and new spike times larger than trial duration were wrapped around to the begining of the trial
    # This process was repeated 100 times.
    camTimes = params['camTimes']
    if not('repeat' in params.keys()):
        params['repeat'] = 100     # shuffling times for calculating speed score threshold, default=100
    if not('minimum_shift' in params.keys()):
        params['minimum_shift'] = -0.5  # the minimum shift time, default=0.5s
    if not('maximum_shift' in params.keys()):
        params['maximum_shift'] = 0.5
    if not('window' in params.keys()):
        params['window'] = camTimes   # the ephys recording window, default=camTimes, aligned to behaviors

    lf_r_shuffle = np.zeros([st_mtx.shape[0],params['repeat']])
    lr_r_shuffle = np.zeros_like(lf_r_shuffle)
    rf_r_shuffle = np.zeros_like(lf_r_shuffle)
    rr_r_shuffle = np.zeros_like(lf_r_shuffle)
    lf_psi_p = np.zeros(st_mtx.shape[0])
    lr_psi_p = np.zeros_like(lf_psi_p)
    rf_psi_p = np.zeros_like(lf_psi_p)
    rr_psi_p = np.zeros_like(lf_psi_p)
    rval_shuffle = np.zeros([st_mtx.shape[0],params['repeat']])
    speed_score_p = np.zeros(st_mtx.shape[0])
    start_shuffle = np.zeros([st_mtx.shape[0],params['repeat']])
    stop_shuffle = np.zeros([st_mtx.shape[0],params['repeat']])
    rng = np.random.default_rng()
    for i in range(params['repeat']):
        # shuffle spike times
        st_mtx_shuffled = shuffle_spike_times_full(st_mtx, params)
        ##1 calculate PHASE index with shuffled data
        walk_spike_angle_mtx_shuffled = get4limb_spa_stride(angle, st_mtx_shuffled, stride)
        temp = get4limb_spa_circ_stride(walk_spike_angle_mtx_shuffled, stride_h, params)
        lf_r_shuffle[:,i] = temp['lf'][:,0]
        lr_r_shuffle[:,i] = temp['lr'][:,0]
        rf_r_shuffle[:,i] = temp['rf'][:,0]
        rr_r_shuffle[:,i] = temp['rr'][:,0]
        del temp
        ##2 calculate SPEED index with shuffled binning firing rate
        for j in range(binning_fr.shape[1]):
            rval_shuffle[j,i],_ = pearsonr(binning_speed, \
                                           rng.permuted(binning_fr[:,j]))
    #calculate the P value of phase-locking & speed
    for i in range(st_mtx.shape[0]):
        lf_psi_p[i] = sum(j > psi['lf'][i,0] for j in lf_r_shuffle[i,:])/params['repeat']
        lr_psi_p[i] = sum(j > psi['lr'][i,0] for j in lr_r_shuffle[i,:])/params['repeat']
        rf_psi_p[i] = sum(j > psi['rf'][i,0] for j in rf_r_shuffle[i,:])/params['repeat']
        rr_psi_p[i] = sum(j > psi['rr'][i,0] for j in rr_r_shuffle[i,:])/params['repeat']
        speed_score_p[i] = sum(abs(j) > abs(rval[i]) for j in rval_shuffle[i,:])/params['repeat']
    # SAVE data into dataframe
    psi_shuffled = {'lf_shuffle': lf_r_shuffle, 'lr_shuffle': lr_r_shuffle,\
                    'rf_shuffle': rf_r_shuffle, 'rr_shuffle': rr_r_shuffle,\
                    'lf_p': lf_psi_p, 'lr_p': lr_psi_p, 'rf_p': rf_psi_p,\
                    'rr_p': rr_psi_p, 'rval': rval,'pval':pval,'rval_shuffle': rval_shuffle,\
                    'speed_score_p': speed_score_p, 'start_p': start_p,\
                    'start_shuffle': start_shuffle, 'start_score_p': start_p,\
                    'stop_shuffle': stop_shuffle, 'stop_score_p': stop_p,\
                    'stop_p': stop_p,'params': params}
    
    psi.update(psi_shuffled)
    return psi

def get_stride_df(psi, tagging, xcoords, ycoords, unitBestCh,\
               mouse_id, group_id, strain, stage, **kwargs):
    '''
    put all single limb walking phase into one dict
    '''
    lf_psi = psi['lf']
    row_size = lf_psi.shape[0]*4
    depth = -1*ycoords[unitBestCh-1]+3300 #depth of each unit,relative to bregma
    x = xcoords[unitBestCh-1]
    if 'kw' in kwargs.keys():
        lat = kwargs['kw']['latency'].tolist()
        bafr = kwargs['kw']['baseline_fr'].tolist()
        phase_data = {
            'mouse_id': [mouse_id for x in range(row_size)],
            'group_id': [group_id for x in range(row_size)],
            'strain': [strain for x in range(row_size)],
            'stage': [stage for x in range(row_size)],
            'limb': ["lf" for x in range(len(lf_psi))]+["lr" for x in range(len(lf_psi))]+\
            ["rf" for x in range(len(lf_psi))]+["rr" for x in range(len(lf_psi))],
            'tagging': tagging.tolist()+tagging.tolist()+tagging.tolist()+tagging.tolist(),
            'depth': depth.tolist()+depth.tolist()+depth.tolist()+depth.tolist(),
            'x': x.tolist()+x.tolist()+x.tolist()+x.tolist(),
            'bestCh':unitBestCh.tolist()+unitBestCh.tolist()+unitBestCh.tolist()+unitBestCh.tolist(),
            'r': psi['lf'][:,0].tolist()+psi['lr'][:,0].tolist()+psi['rf'][:,0].tolist()+\
            psi['rr'][:,0].tolist(),
            'a': psi['lf'][:,1].tolist()+psi['lr'][:,1].tolist()+psi['rf'][:,1].tolist()+\
            psi['rr'][:,1].tolist(),
            'rayleigh_p': psi['lf'][:,2].tolist()+psi['lr'][:,2].tolist()+psi['rf'][:,2].tolist()+\
            psi['rr'][:,2].tolist(),
            'p': psi['lf_p'].tolist()+psi['lr_p'].tolist()+psi['rf_p'].tolist()+psi['rr_p'].tolist(),
            'latency':lat + lat + lat + lat,
            'base_fr':bafr + bafr + bafr + bafr,
            'speed_score_p': psi['speed_score_p'].tolist()+psi['speed_score_p'].tolist()+\
                psi['speed_score_p'].tolist()+psi['speed_score_p'].tolist(),
            'start_score_p': psi['start_score_p'].tolist()+psi['start_score_p'].tolist()+\
                psi['start_score_p'].tolist()+psi['start_score_p'].tolist(),
            'stop_score_p': psi['stop_score_p'].tolist()+psi['stop_score_p'].tolist()+\
                psi['stop_score_p'].tolist()+psi['stop_score_p'].tolist(),
            'pval': psi['pval'].tolist()+psi['pval'].tolist()+psi['pval'].tolist()+\
                psi['pval'].tolist()
            }
    else:
        phase_data = {
            'mouse_id': [mouse_id for x in range(row_size)],
            'group_id': [group_id for x in range(row_size)],
            'strain': [strain for x in range(row_size)],
            'stage': [stage for x in range(row_size)],
            'limb': ["lf" for x in range(len(lf_psi))]+["lr" for x in range(len(lf_psi))]+\
            ["rf" for x in range(len(lf_psi))]+["rr" for x in range(len(lf_psi))],
            'tagging': tagging.tolist()+tagging.tolist()+tagging.tolist()+tagging.tolist(),
            'depth': depth.tolist()+depth.tolist()+depth.tolist()+depth.tolist(),
            'x': x.tolist()+x.tolist()+x.tolist()+x.tolist(),
            'bestCh':unitBestCh.tolist()+unitBestCh.tolist()+unitBestCh.tolist()+unitBestCh.tolist(),
            'r': psi['lf'][:,0].tolist()+psi['lr'][:,0].tolist()+psi['rf'][:,0].tolist()+\
            psi['rr'][:,0].tolist(),
            'a': psi['lf'][:,1].tolist()+psi['lr'][:,1].tolist()+psi['rf'][:,1].tolist()+\
            psi['rr'][:,1].tolist(),
            'rayleigh_p': psi['lf'][:,2].tolist()+psi['lr'][:,2].tolist()+psi['rf'][:,2].tolist()+\
            psi['rr'][:,2].tolist(),
            'p': psi['lf_p'].tolist()+psi['lr_p'].tolist()+psi['rf_p'].tolist()+psi['rr_p'].tolist(),
            'speed_score_p': psi['speed_score_p'].tolist()+psi['speed_score_p'].tolist()+\
                psi['speed_score_p'].tolist()+psi['speed_score_p'].tolist(),
            'start_score_p': psi['start_score_p'].tolist()+psi['start_score_p'].tolist()+\
                psi['start_score_p'].tolist()+psi['start_score_p'].tolist(),
            'stop_score_p': psi['stop_score_p'].tolist()+psi['stop_score_p'].tolist()+\
                psi['stop_score_p'].tolist()+psi['stop_score_p'].tolist(),
            'pval': psi['pval'].tolist()+psi['pval'].tolist()+psi['pval'].tolist()+\
                psi['pval'].tolist()
        }
    
    df = pd.DataFrame(data=phase_data)
    
    return df

def get_full_stride(stride, walkTimes):
    '''
    find full stride: use left rear paw's stance start point as a reference, before next 
    stance start point, if all other 3 paws (lf, rf, rr) are detected, this stride is called
    a full stride, which is registered for further analysis.
    
    input: stride (dict) - including 4 limbs' stride (lf, lr, rf, rr), each limb including
            'stride' (tuple) - 3 times (stance-swing_stance)
            'stride-flag' (list) - stride-walking mapping
    output: full_stride (list) - including all detected full strides, arranged in sequence:
            lr_stance, lr_swing, lr_stance, lf_stance, lf_swing, lf_stance, rf_stance, rf_swing,
            rf_stance, rr_stance, rr_swing, rr_stance, ...
            every 12 elements make a full stride.
            full_stride_flag (list) - save the corresponding walking index
    updated: 11/6/2022, add flag to each full stride, mapping with specific walking bout
    '''
    lf, lr, rf, rr = stride['lf']['stride'], stride['lr']['stride'], stride['rf']['stride'], stride['rr']['stride']
    lf_flag= stride['lf']['stride_flag'].tolist()
    lr_flag= stride['lr']['stride_flag'].tolist()
    rf_flag= stride['rf']['stride_flag'].tolist()
    rr_flag= stride['rr']['stride_flag'].tolist()
    full_stride = []
    full_stride_flag = []
    for i in range(walkTimes.shape[0]):
        lf_n = lf_flag.count(i)
        lr_n = lr_flag.count(i)
        rf_n = rf_flag.count(i)
        rr_n = rr_flag.count(i)
        if lf_n!=0 and lr_n!=0 and rf_n!=0 and rr_n!=0:
            for j in range(lr_n):
                lr_stride = lr[j+lr_flag.index(i)].tolist()
                temp_stance_phase = lr_stride
                k1=0
                while k1<lf_n:
                    lf_stride = lf[k1+lf_flag.index(i)].tolist()
                    if lf_stride[0]>lr_stride[0] and lf_stride[0]<lr_stride[2]:
                        temp_stance_phase = temp_stance_phase + lf_stride
                        k2=0
                        while k2<rf_n:
                            rf_stride = rf[k2+rf_flag.index(i)].tolist()
                            if rf_stride[0]>lr_stride[0] and rf_stride[0]<lr_stride[2]:
                                temp_stance_phase = temp_stance_phase + rf_stride
                                k3=0
                                while k3<rr_n:
                                    rr_stride = rr[k3+rr_flag.index(i)].tolist()
                                    if rr_stride[0]>lr_stride[0] and rr_stride[0]<lr_stride[2]:
                                        temp_stance_phase = temp_stance_phase + rr_stride
                                        break
                                    k3 = k3+1
                                break
                            k2 = k2+1
                        break
                    k1 = k1+1
                if len(temp_stance_phase)==12:
                    full_stride = full_stride + temp_stance_phase
                    full_stride_flag = full_stride_flag + [i]*12

    return full_stride, full_stride_flag

def restrict_spike_times(st, event, win):
    event_spike_times = []
    interval = []
    for t in event:
        interval.append(win + t)
    interval = np.array(interval, object)
    
    for i in range(interval.shape[0]):
        event_mask = (st >= interval[i][0]) & (st < interval[i][1])
        event_spike_times.append(st[event_mask] - event[i])
    
    return np.array(event_spike_times, object)

def bins_spike_times(spike_times,bs,win):
    
    bins = np.arange(win[0],win[1],bs)
    sp_psth = np.zeros(len(bins))
    
    if spike_times.shape[0] > 1: # multi trial case
        bins_spikes = np.zeros((spike_times.shape[0],len(bins)-1))
        for i in range(spike_times.shape[0]):
            bins_spikes[i,:], _ = np.histogram(spike_times[i], bins)

        bins_spikes = bins_spikes/bs #firing rate
        sp_psth = np.mean(bins_spikes, axis=0)
        sp_psth_sem = sem(bins_spikes, axis=0)
        #sp_psth_sd = np.std(bins_spikes, axis=0)
        #sp_psth_sem = sp_psth_sd/np.sqrt(bins_spikes.shape[0])
    else: # single trial
        sp_psth, _ = np.histogram(spike_times, bins)
        sp_psth = sp_psth/bs
        bins_spikes = sp_psth
        sp_psth_sem = np.zeros_like(sp_psth)
    t = bins[:-1]
    
    #if len(sp_psth)==nbins:
    #    sp_psth=sp_psth[:-1]
    #    t = t[:-1]
    return t, sp_psth, sp_psth_sem, bins_spikes

# local function used by mouse_kinematics
def speedMeter(data, fs, params):
    '''
    calculate the speed, acceleration and start/stop times
    5/17/2023 - update start/stop times calculation
    1. body speed > 50 mm/s
    2. start stop duration > 0.5s
    3. for start times, go backwards to the first time point body speed cross 20mm/s;
    for stop times, go foreward to the first time point body speed cross 20mm/s.
    
    Parameters
    ----------
    data : numpy array
        spatial location - 1st-col x-coordination; 2nd-col y-coordination
    fs : constant
        sampling rate
    params : dict
        start_high - high speed threshold for start/stop detection
        start_low - low speed threshold for start/stop detection
        start_stop_minimum - minimum duration for start/stop detection

    Returns
    -------
    sp : list
        speed
    start_stop : numpy array [Mx2]
        start/stop times, M is the total start/stop walking pairs;
        1st-col is the start times; 2nd-col is the paired stop times
    acce : list
        acceleration

    '''
    # calculate speed
    displacement = np.diff(data, axis=0)
    sp = np.sqrt(np.square(displacement[:,0]) + np.square(displacement[:,1])) * fs
    #1/25/2022, use savgol_filter
    #sp = smooth_diff(data) * fs
    
    # calculate acceleration
    speed_incre = np.diff(sp)
    acce = speed_incre * fs
    
    # calculate speed up times, set a simple threshold here, may not good
    sptimes = np.arange(0, sp.shape[0]/fs, 1/fs)
    temp_high = np.zeros(sp.shape)
    temp_high[np.where(sp > params['start_high'])[0]] = 1

    high_diff = np.diff(temp_high)
    sp_up_h = sptimes[np.where(high_diff == 1)]
    sp_down_h = sptimes[np.where(high_diff == -1)]
    sp_up_h = sp_up_h[np.where(sp_up_h>params['camTimes'][0])[0]]
    sp_down_h = sp_down_h[np.where(sp_down_h > sp_up_h[0])[0]] # remove beginning error
    sp_up_h = sp_up_h[np.where(sp_up_h < sp_down_h[-1])[0]] # remove the end error
    start_stop_h = np.concatenate((np.expand_dims(sp_up_h,axis=1), \
                                  np.expand_dims(sp_down_h, axis=1)), axis=1)
    # remove start/stop pairs with a duration shorter than 500ms
    short_index = np.where((start_stop_h[:,1]-start_stop_h[:,0])<\
                           params['start_stop_minimum'])[0]
    start_stop_h = np.delete(start_stop_h,short_index,0)

    temp_low = np.zeros(sp.shape)
    temp_low[np.where(sp > params['start_low'])] = 1 #set a lower threshold for start detection
    low_diff = np.diff(temp_low)
    sp_up_l = sptimes[np.where(low_diff==1)]
    sp_down_l = sptimes[np.where(low_diff==-1)]
    sp_up_l = sp_up_l[np.where(sp_up_l>params['camTimes'][0])[0]]
    sp_down_l = sp_down_l[np.where(sp_down_l > sp_up_l[0])[0]] # remove beginning error
    sp_up_l = sp_up_l[np.where(sp_up_l < sp_down_l[-1])[0]] # remove the end error
    
    start_stop = np.zeros_like(start_stop_h)
    error_index = []
    for i in range(start_stop_h.shape[0]):
        start = start_stop_h[i,0]
        stop = start_stop_h[i,1]
        ind1 = np.where(sp_up_l<start)[0]
        ind2 = np.where(sp_down_l>stop)[0]
        if len(ind1)>0 and len(ind2)>0:
            start_stop[i,0] = sp_up_l[ind1[-1]]
            start_stop[i,1] = sp_down_l[ind2[0]]
        else:
            error_index.append(i)
    if error_index:
        start_stop = np.delete(start_stop,error_index,0)
    start_stop = np.unique(start_stop, axis=0)

    return sp, start_stop, acce

def mouse_kinematics(mouse, params):
    
    if not('spds_n' in params.keys()):
        params['spds_n'] = 1        # default downsampling size 1, which means no downsampling
    if not('start_high' in params.keys()):
        params['start_high'] = 50        # default high threshold of initiation detection, 50mm/s
    if not('start_low' in params.keys()):
        params['start_low'] = 20         # default low threshold of initiation detection, 20mm/s
    if not('start_stop_minimum' in params.keys()):
        params['start_stop_minimum'] = 0.3 # default minimum duration 0.5s 
    fs = params['fr']/params['spds_n']
    kinematics = {}
    
    # calculate body centroid speed
    centroidMouse = mouse['body']
    centroidMouse_ds = centroidMouse[::params['spds_n'],0:2] # downsample data
    bd_speed, bd_speedUpTimes, bd_acce = speedMeter(centroidMouse_ds,fs,params) 
    
    # calculate left forepaw speed
    lf_ds = mouse['lf'][::params['spds_n'],0:2]
    lf_speed, lf_speedUpTimes, lf_acce = speedMeter(lf_ds,fs,params)
    
    # calculate left rearpaw speed
    lr_ds = mouse['lr'][::params['spds_n'],0:2]
    lr_speed, lr_speedUpTimes, lr_acce = speedMeter(lr_ds,fs,params)
    
    # calculate right forepaw speed
    rf_ds = mouse['rf'][::params['spds_n'],0:2]
    rf_speed, rf_speedUpTimes, rf_acce = speedMeter(rf_ds,fs,params)
    
    # calculate right rearpaw speed
    rr_ds = mouse['rr'][::params['spds_n'],0:2]
    rr_speed, rr_speedUpTimes, rr_acce = speedMeter(rr_ds,fs,params)
    
    # calculate nose speed
    nose_ds = mouse['nose'][::params['spds_n'],0:2]
    nose_speed, nose_speedUpTimes, nose_acce = speedMeter(nose_ds,fs,params)
    
    # calculate nose speed
    tail_ds = mouse['tail'][::params['spds_n'],0:2]
    tail_speed, tail_speedUpTimes, tail_acce = speedMeter(tail_ds,fs,params)
    
    kinematics['bd_speed'],kinematics['bd_start_stop'],kinematics['bd_acce'] = bd_speed,bd_speedUpTimes,bd_acce
    kinematics['lf_speed'],kinematics['lf_start_stop'],kinematics['lf_acce'] = lf_speed,lf_speedUpTimes,lf_acce
    kinematics['lr_speed'],kinematics['lr_start_stop'],kinematics['lr_acce'] = lr_speed,lr_speedUpTimes,lr_acce
    kinematics['rf_speed'],kinematics['rf_start_stop'],kinematics['rf_acce'] = rf_speed,rf_speedUpTimes,rf_acce
    kinematics['rr_speed'],kinematics['rr_start_stop'],kinematics['rr_acce'] = rr_speed,rr_speedUpTimes,rr_acce
    kinematics['nose_speed'],kinematics['nose_start_stop'],kinematics['nose_acce'] = nose_speed,nose_speedUpTimes,nose_acce
    kinematics['tail_speed'],kinematics['tail_start_stop'],kinematics['tail_acce'] = tail_speed,tail_speedUpTimes,tail_acce
    kinematics['params'] = params
    
    return kinematics

from scipy.stats import ttest_rel
def psth_norm(event, st_mtx, tagging, params):
    if not('base_win' in params.keys()):
        params['base_win'] = [-5,-1]
    if not('sig_win' in params.keys()):
        params['sig_win'] = [-0.5,0.5]
    if not('binsize' in params.keys()):
        params['binsize'] = 0.02 # bin size 20ms
    if not('start_stop_win' in params.keys()):
        params['start_stop_win'] = [-10, 10]
    window = params['start_stop_win']
    binsize = params['binsize'] 
    sm_n=3
    base_win = params['base_win']
    sig_win = params['sig_win']
    #sig_nb = 25 # compare 0.5s pre/post for significance
    
    n_tagging = len(np.where(tagging == 1)[0])
    psth1_sm = np.zeros([n_tagging, int((window[1]-window[0])/binsize-1)])
    psth1_se = np.zeros([n_tagging, int((window[1]-window[0])/binsize-1)])
    psth1_p = np.zeros(n_tagging)
    psth1_t = np.zeros(n_tagging)
    i = 0
    
    for cluster_id in range(st_mtx.shape[0]):
        if tagging[cluster_id] == 1:
            sp = st_mtx[cluster_id]
            event1_spike_times = \
                restrict_spike_times(st = sp, event = event, win=window)
            t, psth1,psth1_se[i,:],psth1_mtx = \
                bins_spike_times(spike_times = event1_spike_times, \
                                 bs = binsize, win = window)
            psth1_sm[i,:] = gaussian_filter(psth1, sigma = sm_n)
            
            # t test
            ind1 = int((base_win[0]-window[0])/binsize)
            ind2 = int((base_win[1]-window[0])/binsize)
            pre_mean = np.mean(psth1_mtx[:,ind1:ind2],axis=1)
            ind1 = int((sig_win[0]-window[0])/binsize)
            ind2 = int((sig_win[1]-window[0])/binsize)
            post_mean = np.mean(psth1_mtx[:,ind1:ind2],axis=1)
            psth1_p[i] = ttest_rel(pre_mean, post_mean).pvalue
            psth1_t[i] = ttest_rel(pre_mean, post_mean).statistic
            i = i+1
    
    # normalization
    #pre_walk = np.mean(psth1_sm[:,0:499],axis=1)
    #psth1_sm = psth1_sm/np.tile(np.expand_dims(pre_walk,axis=1),psth1_sm.shape[1])
    
    return t, psth1_sm, psth1_se, psth1_p, psth1_t

def calculate_speed_score(speed, spike_time, params):
    '''calculate speed score (ss),defined as the pearson product-moment 
    correlation between instantaneous (only works for linear speed encoding)
    firing rate and speed.
    ref: Kropff, E.et al, Nature 523, 419–424 (2015).
    1. calculate firing rate
    2. bin the speed
    2. remove bad bins
    4. calcualte pearson correlation'''
    
    #ss = np.zeros((1,spike_time.shape[0])) # speed score
    if not('speed_space' in params.keys()):
        params['speed_space'] = 10    # default speed bin size 10mm/s
    if not('speed_bin_valid_th' in params.keys()):
        params['speed_bin_valid_th'] = 0.005 # default at least 0.5% data included
    if not('maximum_speed' in params.keys()):
        params['maximum_speed'] = 300
    if not('remove_bad_bins' in params.keys()):
        params['remove_bad_bins'] = False
    if not('minimum_speed' in params.keys()):
        params['minimum_speed'] = 0
        
    # calculate firing rate, binsize equal to single frame duration
    fr = np.zeros((speed.shape[0], spike_time.shape[0]))
    nbins = speed.shape[0]
    win = params['window']
    bins = np.arange(win[0],win[1]+(win[1]-win[0])/nbins,\
                     (win[1]-win[0])/nbins)
        
    for i in range(spike_time.shape[0]): #loop through neurons
        spt = spike_time[i]
        sp_psth, _ = np.histogram(spt, bins)
        sp_psth = sp_psth/((win[1]-win[0])/nbins)
        fr[:,i] = gaussian_filter(sp_psth, sigma = 3)
        del sp_psth
        
    #speed = speed[:-1] # bins_spike_times removed the last element for firing rate
    
    speed_space = params['speed_space'] # default 10mm/s
    bin_check_th = params['speed_bin_valid_th'] # default 0.5% info included
    max_speed = params['maximum_speed']
    speed_bin = int(max_speed/speed_space) #default maximum speed=400mm/s
    bin_sample_size = np.zeros(speed_bin)
    binning_speed = np.zeros(speed_bin)
    binning_fr = np.zeros((speed_bin,fr.shape[1]))
    binning_se = np.zeros((speed_bin, fr.shape[1]))
    for i in range(speed_bin):
        index = np.intersect1d(np.where(speed >= speed_space*i)[0],\
                               np.where(speed < speed_space*(i+1))[0])
        if index.shape[0]>0:
            bin_sample_size[i] = index.shape[0]
            #binning_speed[i,0] = np.mean(speed[index])
            binning_speed[i] = speed_space*i
            binning_fr[i,:] = np.mean(fr[index,:], axis=0)
            binning_se[i,:] = sem(fr[index,:], axis=0)
    # only bins accounting for at least 0.5% of data included
    if params['remove_bad_bins']:
        bad_bin_id = np.where(bin_sample_size/speed.shape[0] < bin_check_th)[0] 
        binning_speed = np.delete(binning_speed, bad_bin_id, None)
        binning_fr = np.delete(binning_fr, bad_bin_id, 0)
        binning_se = np.delete(binning_se, bad_bin_id, 0)
    
    if params['minimum_speed']==50:
        # remove first speed bin, which include the rest-movement transition info
        # remove speed < 50mm/s
        binning_speed = binning_speed[5:]
        binning_fr = binning_fr[5:,:]
        binning_se = binning_se[5:,:]
    '''
    # calculate speed score, definition:
    for i in range(spike_time.shape[0]): #loop through all neurons
        ss[0,i], _ = stats.pearsonr(binning_speed, binning_fr[:,i])'''
    return binning_speed, binning_fr, binning_se

def inver_func(x, a, b):
    return b + a/x

def get_tagging(st_mtx, clwf, fs, pulseTimes, params):
    '''
    opto-tagging criteria:
        
    '''
    if not('pre_laser' in params.keys()):
        params['pre_laser'] = 1    # pre-laser firing rate window, 1s
    if not('max_latency' in params.keys()):
        params['max_latency'] = 0.006 # maximum laser response latency 6ms
    if not('min_latency' in params.keys()):
        params['min_latency'] = 0.0006 # minimum laser response latency 0.6ms
    if not('max_base_fr' in params.keys()):
        params['max_base_fr'] = 100 # baseline firing rate < 100Hz
    if not('min_trough_peak' in params.keys()):
        params['min_trough_peak'] = 0.01 # trough peak width > 0.01ms
    if not('trough_ratio' in params.keys()):
        params['trough_ratio'] = 2 # waveform trough amplitude ratio < 2
    if not('min_wf_corr' in params.keys()):
        params['min_wf_corr'] = 0.95 # pre/post laser waveform pearsonr > 0.95
    if not('pre_post_p' in params.keys()):
        params['pre_post_p'] = 0.05 # pre/post laser firing rate paired t-test, p<=0.05
    
    laser_pre_post = np.zeros((len(st_mtx),2)) #1st-col: tagging lable; 2nd-col: pval
    wf_corr = np.zeros((len(st_mtx),2))
    wf_ratio = np.zeros(len(st_mtx))
    base_fr = np.zeros(len(st_mtx))
    trough_peak = np.zeros(len(st_mtx))#spike width
    tagging = np.zeros(len(st_mtx)) 
    mean_latency = np.zeros(len(st_mtx))
    wave_t0 = 34; #default=34. used to select a narrower spike waveform time window.
    wave_tf = 66; #default=66.
    
    for i in range(len(st_mtx)):
        st = st_mtx[i]
        wf = clwf[i]
        n_wf = wf[wave_t0:wave_tf,:]# narrow waveform

        base_fr[i] = len(np.where(st<pulseTimes[0])[0])/pulseTimes[0]
        post_laser_spid = np.empty((1,))
        laser_fr = np.zeros((len(pulseTimes),2))
        latency1st = []
        for j in range(len(pulseTimes)):
            pt = pulseTimes[j]
            pre_index = np.where((st<pt-params['min_latency'])&\
                                 (st>pt-params['pre_laser']))[0]
            post_index = np.where((st>pt+params['min_latency']) &\
                                       (st<pt+params['max_latency']))[0]
            laser_fr[j,0] = len(pre_index)/(params['pre_laser']-params['min_latency'])
            laser_fr[j,1] = len(post_index)/(params['max_latency']-params['min_latency'])
            if len(post_index)>0:
                post_laser_spid = np.append(post_laser_spid,post_index)
                latency1st.append(st[post_index[0]]-pt)
        
        latency1st=np.array(latency1st)
        mean_latency[i] = np.mean(latency1st)
        laser_pre_post[i,1] = ttest_rel(laser_fr[:,0],laser_fr[:,1]).pvalue
        if (laser_pre_post[i,1]<=params['pre_post_p']) and \
            (np.mean(laser_fr[:,1])>np.mean(laser_fr[:,0])):
            laser_pre_post[i,0] = 1
        
        post_laser_spid = np.delete(post_laser_spid,0,None)#remove the 1st element
        if len(post_laser_spid)>0:
            post_laser_spid = post_laser_spid.astype(int)
            post_wf_mean = np.mean(n_wf[:,post_laser_spid],axis=1)
        else:
            post_wf_mean = np.zeros(n_wf.shape[0])
        n_post = len(post_laser_spid)
        base_spid = np.where(st<pulseTimes[0])[0]
        #base_wf_mean = np.mean(wf[:,random.choices(base_spid,k=n_post)],axis=1)
        base_wf_mean = np.mean(n_wf[:,len(base_spid)-n_post+1:-1], axis=1)
        wf_corr[i,:] = pearsonr(post_wf_mean, base_wf_mean)
        
        wf_ratio[i] = min(post_wf_mean)/min(base_wf_mean)
        #calculate trough to peak width from baseline waveform
        trough_id = np.where(base_wf_mean==min(base_wf_mean))[0]
        peak_id = np.where(base_wf_mean[int(trough_id):]==\
                           max(base_wf_mean[int(trough_id):]))[0]
        trough_peak[i] = peak_id/fs*1000 #in ms
        
        # tagging criteria
        tagging_criteria = {
            'pre_post_ttest_pval': laser_pre_post[:,1],
            'wf_corr': wf_corr[:,0],
            'wf_trough_amplitude_ratio': wf_ratio,
            'baseline_fr': base_fr,
            'trough_peak_width': trough_peak,
            'latency': mean_latency}
        tagging_criteria_df = pd.DataFrame(tagging_criteria)
        
        # tagging labels
        if laser_pre_post[i,0]==1 and \
           wf_corr[i,0]>params['min_wf_corr'] and \
           wf_ratio[i]<params['trough_ratio'] and \
           base_fr[i]<params['max_base_fr'] and \
           trough_peak[i]>params['min_trough_peak']:
               tagging[i] = 1
    
    return tagging, tagging_criteria_df

def angle_2vectors(v1, v2):
    
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    
    if abs(dot_product)>1: # in case, unit_v1==unit_v2
        dot_product = 1
    v_angle = np.arccos(dot_product)
    
    c = np.cross(v1,v2)
    if c>0:
        v_angle = v_angle
    else:
        v_angle = -1*v_angle
    return v_angle

def walk_angle_change(v, walk, params):
    if not('minimum_turn' in params.keys()):
        params['minimum_turn'] = 20 # default 20 degree
    walk_angle = []
    walk_mean_angle = np.zeros(walk.shape[0])
    left_right_turn = np.zeros(walk.shape[0])
    for i in range(walk.shape[0]):
        head_angle = []
        if walk[i,1]*params['fr']<v.shape[0]:
            wt = walk[i,:]
            walk_id = np.arange(int(wt[0]*params['fr']), \
                                int(wt[1]*params['fr']),1)
            for j in range(walk_id.shape[0]):
                angle = angle_2vectors(v[walk_id[0],:],v[walk_id[j],:]) # current frame angle
                head_angle = np.append(head_angle, angle)
            head_angle = np.rad2deg(head_angle)
            
            walk_angle.append(head_angle[-1])
            walk_mean_angle[i] = np.mean(head_angle)
            # label left/right turn
            if walk_mean_angle[i]<-1*params['minimum_turn']:
                left_right_turn[i] = -1 # right turn
            elif walk_mean_angle[i]>params['minimum_turn']:
                left_right_turn[i] = 1 # left turn
            else:
                left_right_turn[i] = 0# trivial turn
                
    return walk_angle, left_right_turn

def walk_speed(v, walk, params):
    walk_speed = []
    for i in range(walk.shape[0]):
        wt = walk[i,:]
        start_id = int(np.round(wt[0]*params['fr']))
        stop_id = int(np.round(wt[1]*params['fr']))
        speed = np.mean(v[start_id:stop_id])
        walk_speed = np.append(walk_speed, speed)
    return walk_speed

def walk_distance(coord, walk, params):
    walk_distance = []
    for i in range(walk.shape[0]):
        if walk[i,1]*params['fr']<coord.shape[0]:
            wt = walk[i,:]
            ind1 = int(np.round(wt[0]*params['fr']))
            ind2 = int(np.round(wt[1]*params['fr']))
            pos1 = coord[ind1:ind2,:]
            pos2 = coord[ind1+1:ind2+1,:]
            pos_df = pos2 - pos1
            disance = np.sum(np.sqrt(pos_df[:,0]**2+pos_df[:,1]**2))
            walk_distance = np.append(walk_distance, disance)
    return walk_distance

def limb_coord(mouse, walk, params):
    lf_lr = np.zeros_like(walk) #1st-col: r; 2nd-col: shift time
    lf_rf = np.zeros_like(walk)
    lf_rr = np.zeros_like(walk)
    rf_rr = np.zeros_like(walk)
    rf_lr = np.zeros_like(walk)
    lr_rr = np.zeros_like(walk)
    for i in range(walk.shape[0]):
        wt = walk[i,:]
        start_id = int(np.round(wt[0]*params['fr']))
        stop_id = int(np.round(wt[1]*params['fr']))
        
        lf = mouse['lf'][start_id:stop_id]
        lf = (lf - np.mean(lf))/np.std(lf, ddof=1)
        lr = mouse['lr'][start_id:stop_id]
        lr = (lr - np.mean(lr))/np.std(lr, ddof=1)
        rf = mouse['rf'][start_id:stop_id]
        rf = (rf - np.mean(rf))/np.std(rf, ddof=1)
        rr = mouse['rr'][start_id:stop_id]
        rr = (rr - np.mean(rr))/np.std(rr, ddof=1)
        
        lf_lr[i,:] = pearsonr(lf, lr)
        #temp = np.max(r_lf_lr) #maximum correlation coefficients
        #lf_lr[i,0] = np.arctanh(temp)#0.5*(np.log((1+temp)/(1-temp)))
        #lf_lr[i,1] = np.abs(np.where(r_lf_lr==temp)[0]-len(lf))/params['fr'] #optimal shift time
        lf_rf[i,:] = pearsonr(lf, rf)
        #temp = np.max(r_lf_rf) #maximum correlation coefficients
        #lf_rf[i,0] = np.arctanh(temp)#0.5*(np.log((1+temp)/(1-temp)))
        #lf_rf[i,1] = np.abs(np.where(r_lf_rf==temp)[0]-len(lf))/params['fr'] #optimal shift time
        lf_rr[i,:] = pearsonr(lf, rr)
        #temp = np.max(r_lf_rr) #maximum correlation coefficients
        #lf_rr[i,0] = np.arctanh(temp)#0.5*(np.log(1+temp)/(1-temp))
        #lf_rr[i,1] = np.abs(np.where(r_lf_rr==temp)[0]-len(lf))/params['fr'] #optimal shift time
        rf_rr[i,:] = pearsonr(rf, rr)
        rf_lr[i,:] = pearsonr(rf, lr)
        lr_rr[i,:] = pearsonr(lr, rr)
    # Fisher z transform
    #lf_lr[:,0] = 0.5*(np.log((1+lf_lr[:,0])/(1-lf_lr[:,0])))
    #lf_rf[:,0] = 0.5*(np.log((1+lf_rf[:,0])/(1-lf_rf[:,0])))
   # lf_rr[:,0] = 0.5*(np.log((1+lf_rr[:,0])/(1-lf_rr[:,0])))
    return lf_lr[:,0], lf_rf[:,0], lf_rr[:,0], rf_rr[:,0], rf_lr[:,0], lr_rr[:,0]

def get_healthy_df(speed_r, speed_p, start_p, stop_p, mouse_id, strain):
    '''
    put all neurons' speed fr into one dict
    '''
    row_size = speed_r.shape[0]

    hl_dict = {
        'mouse_id': [mouse_id for x in range(row_size)],
        'strain': [strain for x in range(row_size)],
        'pearsonr': speed_r.tolist(),
        'pearsonp': speed_p.tolist(),
        'start_p': start_p.tolist(),
        'stop_p': stop_p.tolist()
        }
    
    hl_dict_df = pd.DataFrame(data=hl_dict)
    
    return hl_dict_df

def get_gait_df(gait, mouse_id, group_id, strain, stage):
    '''
    put all single limb walking phase into one dict
    '''
    lf, lr, rf, rr = gait['lf'], gait['lr'], gait['rf'], gait['rr']
    row_size = len(lf)+len(lr)+len(rf)+len(rr)

    phase_data = {
        'mouse_id': [mouse_id for x in range(row_size)],
        'group_id': [group_id for x in range(row_size)],
        'strain': [strain for x in range(row_size)],
        'stage': [stage for x in range(row_size)],
        'limb': ["lf" for x in range(len(lf))]+["lr" for x in range(len(lr))]+\
        ["rf" for x in range(len(rf))]+["rr" for x in range(len(rr))],
        'stride_velocity': lf['stride_velocity'].tolist()+lr['stride_velocity'].tolist()+\
            rf['stride_velocity'].tolist()+rr['stride_velocity'].tolist(),
        'stride_length': lf['stride_length'].tolist()+lr['stride_length'].tolist()+\
            rf['stride_length'].tolist()+rr['stride_length'].tolist(),
        'cadence': lf['cadence'].tolist()+lr['cadence'].tolist()+rf['cadence'].tolist()+rr['cadence'].tolist(),
        'swing_stance_ratio': lf['swing_stance_ratio'].tolist()+lr['swing_stance_ratio'].tolist()+\
            rf['swing_stance_ratio'].tolist()+rr['swing_stance_ratio'].tolist(),
        'swing_duration': lf['swing_duration'].tolist()+lr['swing_duration'].tolist()+\
            rf['swing_duration'].tolist()+rr['swing_duration'].tolist(),
        'stance_duration': lf['stance_duration'].tolist()+lr['stance_duration'].tolist()+\
            rf['stance_duration'].tolist()+rr['stance_duration'].tolist(),
        'swing_length': lf['swing_length'].tolist()+lr['swing_length'].tolist()+\
            rf['swing_length'].tolist()+rr['swing_length'].tolist(),
        }
    
    df = pd.DataFrame(data=phase_data)
    
    return df
