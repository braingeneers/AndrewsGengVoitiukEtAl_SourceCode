import networkx as nx
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from umap import umap_ as umap
import cylouvain
import shap
import igraph as ig
# from .autonotebook import tqdm as notebook_tqdm

import zipfile
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
# import seaborn as sns
from scipy import interpolate
from scipy import io
import pickle as pkl
from scipy import ndimage
# import h5py
import xml.etree.ElementTree as ET


# parameters
# CUSTOM_PAL_SORT_3 = ['#5e60ce', '#00c49a', '#ffca3a', '#D81159', '#fe7f2d', '#7bdff2', '#0496ff', '#efa6c9', '#ced4da',
#                      '#9c27b0', '#673ab7', '#009688', '#cddc39', '#8bc34a']
CUSTOM_PAL_SORT_3 = ['#5e60ce', '#00c49a', '#ffca3a', '#D81159', '#fe7f2d', '#7bdff2', '#0496ff', '#efa6c9', '#ced4da',
                     '#9c27b0', '#673ab7', '#009688', '#cddc39', '#8bc34a',
                     '#f06292', '#2196f3', '#ff5722', '#4caf50', '#ffc107', '#795548', '#607d8b', '#8e24aa', '#03a9f4',
                     '#ff9800', '#9e9e9e', '#e91e63', '#3f51b5', '#00bcd4', '#ffeb3b', '#4caf50', '#ff5252', '#03a9f4',
                     '#8bc34a', '#9c27b0']
REC_COLOR = ['#f06292', '#2196f3', '#ff5722', '#4caf50', '#ffc107', '#795548', '#607d8b', '#8e24aa', '#03a9f4',
             '#ff9800', '#9e9e9e', '#e91e63', '#3f51b5', '#00bcd4', '#ffeb3b']

N_NEIGHBORS = 20
MIN_DIST = 0.1
RAND_STATE = 42
# RESOLUTION = 1.5
# BLUE COLOR
BlueCol = '\033[94m'
np.random.seed(RAND_STATE)
os.environ['PYTHONHASHSEED'] = str(RAND_STATE)
random.seed(RAND_STATE)
sns.palplot(CUSTOM_PAL_SORT_3)


# functions 
def load_curation(qm_path):
    with zipfile.ZipFile(qm_path, 'r') as f_zip:
        qm = f_zip.open("qm.npz")
        data = np.load(qm, allow_pickle=True)
        spike_times = data["train"].item()
        fs = data["fs"]
        train = [times / fs for _, times in spike_times.items()]
        if "config" in data:
            config = data["config"].item()
        else:
            config = None
        neuron_data = data["neuron_data"].item()
    return train, neuron_data, config, fs

def remove_single_channel_unit(trains, neuron_data, nelec=2, pitch=17.5):
    assert len(trains) == len(neuron_data), "spike trains and neuron_data must have the same length"
    num = len(neuron_data)
    to_remove = []
    cleaned_data = {}
    cleaned_trains = []
    new_key = 0
    for i, data in neuron_data.items():
        nb_pos = data["neighbor_positions"]
        distance = np.array([np.round(abs(np.linalg.norm(pos-nb_pos[0])), 4) for pos in nb_pos])
        min_dist = np.round(np.sqrt(2*(nelec*pitch)**2), 4)
        if np.sum(distance <= min_dist) == 1:
            to_remove.append(i)
        else:
            cleaned_trains.append(trains[i])
            cleaned_data[new_key] = data
            new_key += 1  
    to_keep = np.setdiff1d(np.arange(num), to_remove)   # to_keep is the old key of the neuron_data
    return to_keep, cleaned_trains, cleaned_data

def waveform_feature(waveform, mid=20, right_only=True, fs=20000.0):
    """
    measure the waveform features for both positive 
    and negative spikes.
    waveform: a list or array of waveform
    mid: the index of the trough
    fs: sampling frequency
    return:
        a dictionary of features including 
        left peak (index, value), 
        right peak (index, value), 
        trough to peak time (time in ms), 
        fwhm (full width half maximum value in ms), 
        fwhm_x (the left and right x values of the fwhm)
    """
    # 0. flip the waveform if it is positive
    if waveform[mid] > 0:
        waveform = -waveform
    # 1. measure the left and the right peak
    left = waveform[:mid+1][::-1]
    right = waveform[mid:]
    ldiff = left[1:] - left[:-1]
    rdiff = right[1:] - right[:-1]
    lpeak_ind_list = np.where(ldiff < 0)[0]
    lpeak_ind, rpeak_ind = 0, len(waveform)-1
    for ind in lpeak_ind_list:
        if waveform[mid-ind] > 0:
            lpeak_ind = mid-ind
            break
    lpeak_value = waveform[lpeak_ind]
    rpeak_ind_list = np.where(rdiff < 0)[0]
    for ind in rpeak_ind_list:
        if waveform[mid+ind] > 0:
            rpeak_ind = mid+ind
            break
    rpeak_value = waveform[rpeak_ind]
    lpeak = (lpeak_ind, lpeak_value)
    rpeak = (rpeak_ind, rpeak_value)
    # print(f"left peak {lpeak}, right peak {rpeak}")
    # 2. measure the trough to peak time (peak taken as the max peak either left or right)
    if not right_only:
        post_hyper = lpeak if lpeak_value > rpeak_value else rpeak
    else:
        post_hyper = rpeak
    trough_to_peak = abs(post_hyper[0]-mid)/(fs/1000)   # make it in ms
    # print(f"{trough_to_peak} ms")
    # 3. measure the full width half maximum (FWHM) from the depolarization trough to the baseline  
    half_amp = waveform[mid]/2.0
    # print(f"half amplitude {half_amp}")
    # print(waveform[lpeak_ind: mid])
    # print(waveform[mid: rpeak_ind+1])
    xx_left = np.arange(lpeak_ind, mid+1)
    xx_right = np.arange(mid, rpeak_ind+1)
    # interpolate the left peak to trough waveform to find the half amplitude point
    fl = interpolate.interp1d(waveform[lpeak_ind: mid+1], xx_left)
    fr = interpolate.interp1d(waveform[mid: rpeak_ind+1], xx_right)
    inter_1 = fl(half_amp)
    inter_2 = fr(half_amp)
    fwhm_x = np.array([inter_1, inter_2])
    fwhm = (abs(inter_2)-abs(inter_1))/(fs/1000)
    # print(f"fwhm {fwhm} ms")
    features = {"lpeak": lpeak, "rpeak": rpeak, "trough_to_peak": trough_to_peak, "fwhm": fwhm, "fwhm_x": fwhm_x}
    return features

def moving_average(data, win=5):
    """
    Save function to matlab movmean
    """
    data = np.array(data)
    assert data.ndim == 1, "input must be one-dimension"
    step = np.ceil(win/2).astype(int)
    movmean = np.empty(data.shape)
    i = 0
    while i < movmean.shape[0]:
        for s in range(step, win):
            res = np.mean(data[:s])
            movmean[i] = res
            i += 1
        for s in range(data.shape[0]-i):
            res = np.mean(data[s:s+win])
            movmean[i] = res
            i += 1
    return movmean

def population_rate(trains: list, bin_size=0.1, sigma=10, average=False):
    N = len(trains)
    trains = np.hstack(trains)
    rec_length = np.max(trains)
    bin_num = int(rec_length// bin_size) + 1
    bins = np.linspace(0, rec_length, bin_num)
    fr = np.histogram(trains, bins)[0] / bin_size
    fr_avg = ndimage.gaussian_filter1d(fr, sigma=sigma)
    if average:
        fr_avg /= N
    return bins, fr_avg

def plot_inset(axs, temp_pos, templates, nelec=2, ylim_margin=0, pitch=17.5):
    assert len(temp_pos) == len(templates), "Input length must be the same!"
    # find the max template
    if isinstance(templates, list):
        templates = np.asarray(templates)
    amp = np.max(templates, axis=1) - np.min(templates, axis=1)
    max_amp_index = np.argmax(amp)
    position = temp_pos[max_amp_index]
    axs.scatter(position[0], position[1], linewidth=10, alpha=0.2, color='grey')
    axs.text(position[0], position[1], str(position), color="g", fontsize=12)
    # set same scaling to the insets
    ylim_min = min(templates[max_amp_index])
    ylim_max = max(templates[max_amp_index])
    # choose channels that are close to the center channel
    for i in range(len(temp_pos)):
        chn_pos = temp_pos[i]
        if position[0] - nelec * pitch <= chn_pos[0] <= position[0] + nelec * pitch \
                and position[1] - nelec * pitch <= chn_pos[1] <= position[1] + nelec * pitch:
            axin = axs.inset_axes([chn_pos[0]-5, chn_pos[1]-5, 15, 20], transform=axs.transData)
            axin.plot(templates[i], color='k', linewidth=2, alpha=0.7)
            axin.set_ylim([ylim_min - ylim_margin, ylim_max + ylim_margin])
            axin.set_axis_off()
    axs.set_xlim(position[0]-1.5*nelec*pitch, position[0]+1.5*nelec*pitch)
    axs.set_ylim(position[1]-1.5*nelec*pitch, position[1]+1.5*nelec*pitch)
    axs.invert_yaxis()
    return axs





