# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:39:55 2023

@author: Jeremy
"""

# import packages
import mne
import matplotlib.pyplot as plt 
from scipy.signal import welch, butter, filtfilt, decimate, resample, argrelextrema, argrelmax, find_peaks
import numpy as np 
import pandas as pd 
from utils import quikcap_32, fix_chans
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from scipy.stats import zscore
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
%matplotlib qt

# read raw
fname = "EEG-MRI/real/S2_RAW_BFFE.cdt"
raw_bffe = mne.io.read_raw_curry(fname, preload=True)
fix_chans(raw_bffe)

# sampling rates
fs1 = 10000
fs2 = fs1*2

# Trigger info to get starting time 
trg_slice = raw_bffe.get_data()[raw_bffe.info.ch_names.index('Trigger'), :]
trg = np.where(trg_slice>0.05)[0]
time_trg = np.arange(len(trg)) / fs1


t1 = trg[0]

y = raw_bffe.copy().filter(l_freq=1, 
                          h_freq=None).get_data(picks=['O1'], 
                                                units='uV')[:, t1:]
y = y.flatten()

time_y = np.arange(len(y)) / fs1

y_20k = resample(y, num=(len(y)*2))

time_y_20k = np.arange(len(y_20k)) / fs2

#tr_ref = y_20k[0:39828]
 

# 20k upsampled bFFE for 15 TR:s
y_15trs = y_20k[:597440]
time_y_15trs = np.arange(len(y_15trs)) / fs2

#plt.figure()
#plt.plot(time_y_15trs, y_15trs, '-k')

# Trigger times 20k
trg_times = raw_bffe.get_data()[raw_bffe.info.ch_names.index('Trigger'), t1:]

trg_times_20k = resample(trg_times, num=(len(trg_times)*2))

trg_points_20k = np.where(trg_times_20k>0.06)[0]

trg_points_20k_diff = np.diff(trg_points_20k)

# TR 1 
tr_ref = y_20k[0:39828]

corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

#plt.figure()
#plt.plot(corr_points)

corr_tr_points = np.where(corr_points>42750)[0]

plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 

plt.close()

tr1_times = [0, 79659, 119488, 199147, 238976, 278806, 318635, 398294, 438123, 477953]
tr1_len = 39828

# TR 2 
tr_ref = y_20k[39828:79658]

corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

#plt.figure()
#plt.plot(corr_points)

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 

plt.close()
tr2_times = [39828, 159316, 238975, 358463, 438122, 477951, 557610, 637269, 677098, 756757]
tr2_len = 39830


# TR 3 
tr_ref = y_20k[79658:119488]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')
plt.figure()
plt.plot(corr_points)

corr_tr_points = np.where(corr_points>43000)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.show()

# Using 42885 threshold results in AAS that isn't as good as others
#tr3_times = [79658, 159317, 199146, 278805, 358464, 398293, 477952, 517781, 597440, 677099]

# trying higher threshold 43000
# Careful! This threshold actually misses the original TR. So may not be good
tr3_times = [79658, 199146, 278805, 398293, 477952, 517781, 597440, 716928, 796587, 916075]
tr3_len = 39830



# TR 4 
tr_ref = y_20k[119488:159316]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')
#plt.figure()
#plt.plot(corr_points)

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()
         
tr4_times = [119488, 199147, 238976, 318635, 358464, 398294, 438123, 517782, 557611, 637270]
tr4_len = 39828

# TR 5 
tr_ref = y_20k[159316:199146]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')
#plt.figure()
#plt.plot(corr_points)

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr5_times = [159316, 238975, 278804, 358463, 398292, 438122, 477951, 557610, 597439, 677098]

tr5_len = 39830

# TR 6 
tr_ref = y_20k[199146:238976]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr6_times = [199146, 278805, 318634, 398293, 438122, 477952, 517781, 597440, 637269, 716928]
tr6_len = 39830

# TR 7 
tr_ref = y_20k[238976:278804]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr7_times = [238976, 318635, 358464, 438123, 477952, 517782, 557611, 637270, 677099, 756758]
tr7_len = 39828

# TR 8 
tr_ref = y_20k[278804:318634]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr8_times = [278804, 358463, 398292, 477951, 517780, 557610, 597439, 677098, 716927, 796586]
tr8_len = 39830

# TR 9 
tr_ref = y_20k[318634:358464]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr9_times = [318634, 398293, 438122, 517781, 557610, 597440, 637269, 716928, 756757, 836416]
tr9_len = 39830

# TR 10
tr_ref = y_20k[358464:398292]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr10_times = [358464, 438123, 477952, 557611, 597440, 637270, 677099, 756758, 796587, 876246]
tr10_len = 39828

# TR 11
tr_ref = y_20k[398292:438122]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')
#plt.figure()
#plt.plot(corr_points)

corr_tr_points = np.where(corr_points>43000)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.show()

# using same threshold as other TR's, 42885.
# Resulted in not as good AAS
#tr11_times = [398292, 438121, 477951, 517780, 597439, 637268, 716927, 796586, 836415, 916074]

# using higher threshold, 43000
tr11_times = [398292, 477951, 517780, 597439, 716927, 796586, 916074, 1035562, 1115221, 1433856]
tr11_len = 39830

# TR 12 
tr_ref = y_20k[438122:477952]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr12_times =[438122, 517781, 557610, 637269, 756757, 836416, 876245, 955904, 1035563, 1075392]
tr12_len = 39830

# TR 13 
tr_ref = y_20k[477952:517782]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr13_times = [477952, 557611, 597440, 677099, 796587, 876246, 916075, 995734, 1075393, 1115222]
tr13_len = 39830

# TR 14 
tr_ref = y_20k[517782:557610]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr14_times = [517782, 597441, 637270, 716929, 836417, 916076, 955905, 1035564, 1115223, 1155052]
tr14_len = 39828

# TR 15
tr_ref = y_20k[557610:597440]
corr_points = np.correlate(zscore(y_20k), zscore(tr_ref), mode='valid')

corr_tr_points = np.where(corr_points>42885)[0]
plt.figure()
plt.plot(time_y_20k, y_20k, '-k')
for trigger_time in time_y_20k[corr_tr_points]:
    plt.axvline(trigger_time, color='b', linewidth=1) 
plt.close()

tr15_times = [557610, 637269, 677098, 756757, 876245, 955904, 995733, 1075392, 1155051, 1194880]
tr15_len = 39830

################################

# AAS 
eeg = y_20k.copy()
epochs = 10
clean_data = np.zeros_like(y_15trs)

# TR 1 
tr_starts = tr1_times.copy()
tr_len = tr1_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 2 
tr_starts = tr2_times.copy()
tr_len = tr2_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 3
tr_starts = tr3_times.copy()
tr_len = tr3_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 4
tr_starts = tr4_times.copy()
tr_len = tr4_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 5
tr_starts = tr5_times.copy()
tr_len = tr5_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 6
tr_starts = tr6_times.copy()
tr_len = tr6_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 7 
tr_starts = tr7_times.copy()
tr_len = tr7_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 8
tr_starts = tr8_times.copy()
tr_len = tr8_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 9
tr_starts = tr9_times.copy()
tr_len = tr9_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 10
tr_starts = tr10_times.copy()
tr_len = tr10_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 11
tr_starts = tr11_times.copy()
tr_len = tr11_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 12
tr_starts = tr12_times.copy()
tr_len = tr12_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 13
tr_starts = tr13_times.copy()
tr_len = tr13_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 14
tr_starts = tr14_times.copy()
tr_len = tr14_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

# TR 15
tr_starts = tr15_times.copy()
tr_len = tr15_len

avg_arr = np.nan*np.zeros((epochs, tr_len))

plt.figure()
for i_epoch in range(epochs):
    epoch_length = tr_len
    print("epoch_length: ", epoch_length)
    epoch_start = tr_starts[i_epoch]
    print("epoch_start: ", epoch_start)
    
    epoch = eeg[epoch_start:epoch_start+epoch_length]
    plt.plot(epoch, lw=0.5, label=f'Artifact {i_epoch+1}')
    plt.legend()
    for k, element in enumerate(epoch):
        avg_arr[i_epoch, k] = element


avg = np.nanmean(avg_arr, axis=0)

t_start = tr_starts[0]
print("t_start: ", t_start)
t_end = t_start + tr_len
print("t_end: ", t_end)

print("eeg slice shape: ", eeg[t_start:t_end].shape)

alpha = np.dot(eeg[t_start:t_end], avg)/np.dot(avg, avg)
A_avg_corr = alpha * avg

clean_data[t_start:t_end] = eeg[t_start:t_end] - A_avg_corr

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(eeg[t_start:t_end], '-k')
ax[0].plot(A_avg_corr, '-b')
ax[1].plot(clean_data[t_start:t_end])
plt.show()

##################################

# Final 30s plot 
plt.figure()
plt.plot(time_y_15trs, clean_data, '-k')
plt.show()


# Decimating and filtering 

# 20k -> 10k
bffe_30s_aas_10k = decimate(clean_data, q=2,  ftype='fir')

# 10k -> 1k 
bffe_30s_aas_1k = decimate(bffe_30s_aas_10k, q=10,  ftype='fir')

# LPF 40 Hz 
def bp_filter(data, f_lo, f_hi, fs):
    """Digital band pass filter (6-th order Butterworth)

    Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data
    """
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  # Nyquist frequency
    b_lo = f_lo/f_ny  # normalized frequency [0..1]
    b_hi = f_hi/f_ny  # normalized frequency [0..1]
    # band-pass filter parameters
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    #print(bp_b1, bp_a1)
    #print(bp_b2, bp_a2)
    print(np.sum(np.isnan(data)))
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    #print(np.sum(np.isnan(data_filt)))
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    #print(np.sum(np.isnan(data_filt)))
    return data_filt

bffe_30s_aas_1k_40lpf = bp_filter(bffe_30s_aas_1k, f_lo=1, f_hi=40, fs=1000)
time_bffe_30s_aas_1k_40lpf = np.arange(len(bffe_30s_aas_1k_40lpf)) / 1000


fig, ax = plt.subplots(2, 1, figsize=(12, 6))

ax[0].plot(time_y_15trs, clean_data, label='AAS before dowsnampling + filtering' )
ax[0].set_title('AAS before dowsnampling + filtering')

ax[1].plot(time_bffe_30s_aas_1k_40lpf, bffe_30s_aas_1k_40lpf, '-k', label='Downsampled + 1-40 BPF')
ax[1].set_title('Downsampled + 1-40 BPF')

plt.show()

# ANC attempt 

"""
# Lets create 33 Hz reference 
fs = 20000
tmax = len(clean_data) / fs
n = int(tmax*fs)

x = np.zeros(n)
time = np.arange(n) / fs

x_33 = np.cos(2*np.pi*33*time)

x += x_33

# add noise
x += 0.02*np.random.randn(n)

plt.figure()
plt.plot(time, x)
"""


# Creating artificial "trigger" channel 

#slice_dur_fake = 1.9915 / 65.7195 # from theoretical slice frequency of 33 Hz 

slice_dur_fake = 1.9915 / 66.5161 # from theoretical slice frequency of 33.4 Hz 


fake_slices_times = np.arange(0, len(y_20k), int(fs*slice_dur_fake))

slice_trigger_chan_fake = np.zeros(len(y_20k)) 

for index in fake_slices_times:
    try:
        slice_trigger_chan_fake[index] = 1
    except:
        continue

time_fake = np.arange(len(slice_trigger_chan_fake)) / fs

plt.figure()
plt.plot(time_fake, slice_trigger_chan_fake)
plt.show()

# THIS WORKS!!!!

# decimating + filtering

fake_10k = decimate(slice_trigger_chan_fake, q=2, ftype='fir')
fake_1k = decimate(fake_10k, q=10, ftype='fir')

fake_1k_1to40 = bp_filter(fake_1k, f_lo=1, f_hi=40, fs=1000)


slice_ref_30 = fake_1k_1to40[10000:39872]

slice_ref_30_500x = slice_ref_30 * 500

plt.figure()
plt.plot(slice_ref_30, '-k')
plt.plot(slice_ref_30_500x, '-b')
plt.show()


# ANC
def fastranc(refs, d, N, mu):
    # Reshape input data into column vectors
    refs = refs.reshape(-1, 1)
    d = d.reshape(-1, 1)
    
    # Initialize variables
    mANC = len(d)
    W = np.zeros((N + 1, 1))
    r = np.vstack((np.zeros((1, 1)), refs[0:N]))
    out = np.zeros((mANC, 1))
    y = np.zeros((mANC, 1))

    # Main loop
    for E in range(N, mANC):
        # Update r by shifting and replacing its first element
        r = np.vstack((refs[E], r[0:N]))

        # Calculate filtered output y(E)
        y[E] = np.sum(W * r)

        # Calculate error signal out(E)
        out[E] = d[E] - y[E]

        # Update filter coefficients W using LMS algorithm
        W = W + 2 * mu * out[E] * r

    return out, y

out, y = fastranc(slice_ref_30_500x,
                  bffe_30s_aas_1k_40lpf,
                  N=30,
                  mu=0.001)

out = out.flatten()

time_out = np.arange(len(out)) / 1000

plt.figure()
plt.plot(time_out, bffe_30s_aas_1k_40lpf, '-k', alpha=0.5, label='Original' )
plt.plot(time_out, out, '-b', label='ANC')
plt.plot(time_out, slice_ref_30_500x, '-r', label='Ref')
plt.legend()
plt.show()


# psd to compare
freqs_og, psd_og = welch(bffe_30s_aas_1k_40lpf, fs=1000, nperseg=1024)

freqs_test, psd_test = welch(out, fs=1000, nperseg=1024)

x2 = 42

plt.figure()
plt.semilogy(freqs_og[:x2], psd_og[:x2], '-k', label='Pre-ANC')
plt.semilogy(freqs_test[:x2], psd_test[:x2], '-b', label='Post-ANC')
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Power " + r"$\mu V^2/Hz$", fontsize=12)
plt.margins(x=0)
plt.grid(visible=True)
plt.show()


