# -*- coding: utf-8 -*-

import mne
import matplotlib.pyplot as plt 
from scipy.signal import welch, butter, filtfilt, decimate, resample
from scipy.stats import zscore
import numpy as np 
from utils import fix_chans

%matplotlib qt # Interactive plots in sep. windows


# read raw
fname = "EEG-MRI/real/S2_RAW_EPI.cdt"
raw_epi = mne.io.read_raw_curry(fname, preload=True)
fix_chans(raw_epi)


# Trigger info to get starting time 
trg_slice = raw_epi.get_data()[raw_epi.info.ch_names.index('Trigger'), :]
trg = np.where(trg_slice>0.05)[0]

print(np.diff(trg))

t1 = trg[1]

y = raw_epi.copy().filter(l_freq=1, 
                          h_freq=None).get_data(picks=['O1'], 
                                                units='uV')[:, t1:]
y = y.flatten()
time_y = np.arange(len(y)) / 10000

trgs_y = raw_epi.copy().get_data(picks=['Trigger'])[:, t1:]
trgs_y = trgs_y.flatten()

#plt.plot(trgs_y)

trgs = np.where(trgs_y>0.05)[0]

trgs_diff = np.diff(trgs)


# plot of trigger channel events over EEG
plt.figure()
plt.plot(time_y, y, '-k') 
for trigger_time in time_y[trgs]:
    plt.axvline(trigger_time, color='b', linewidth=1) 

plt.close()


trgs_no_outliers = np.delete(trgs, 34)


print(len(y))
# Interpolation 
y_20k = resample(y, num=(len(y)*2))
time_y_20k = np.arange(len(y_20k)) / 20000


# Slice points creation 
slice_points = np.arange(0, 0 + 1142 * 520, 1142 )

slice_ref = y_20k[0:1142]

# plot of slice_ref for correlation to get accurate slice start times 

plt.figure()
plt.plot(slice_ref, '-k')
plt.close()

y_20k_thirtyseg = y_20k[0:0 + (1142 * 520)]

time_y_20k_thirtyseg = np.arange(len(y_20k_thirtyseg)) / 20000

# plot of 30s GA affected raw eeg 
plt.figure()
plt.plot(time_y_20k_thirtyseg, y_20k_thirtyseg, '-k')
plt.xlabel("Seconds (s)", fontsize=10)
plt.ylabel('Voltage ($\mu$V)', fontsize=10)
plt.tight_layout()
plt.close()

y_20k_90sec = y_20k[0:0 + (1142*1560)]

time_y20k_90sec = np.arange(len(y_20k_90sec)) / 20000 

# Slice points creation 
slice_points = np.arange(0, 0 + 1142 * 520, 1142 )

slice_ref = y_20k[0:1142]

# plot of slice_ref for correlation to get accurate slice start times 
plt.figure()
plt.plot(slice_ref, '-k')
plt.close()

y_20k_thirtyseg = y_20k[0:0 + (1142 * 520)]

time_y_20k_thirtyseg = np.arange(len(y_20k_thirtyseg)) / 20000

# plot of 30s GA affected raw eeg 
plt.figure()
plt.plot(time_y_20k_thirtyseg, y_20k_thirtyseg, '-k')
plt.xlabel("Seconds (s)", fontsize=10)
plt.ylabel('Voltage ($\mu$V)', fontsize=10)
plt.tight_layout()
plt.close()

corr_points = np.correlate(zscore(y_20k), zscore(slice_ref), mode='valid')

# plot of corr_points
plt.figure()
plt.plot(corr_points)

corr_slice_points = np.where(corr_points>1100)[0] # original 1100 # 1170 very good, 1 slice off
#print(corr_slice_points)   

# local maxima conversion of corr_slice_points
local_maxima_indices = []

# Iterate through corr_slice_points
for idx in corr_slice_points:
    # check if first value in given close range
    if len(local_maxima_indices) == 0 or idx - local_maxima_indices[-1] > 5:
        local_maxima_indices.append(idx)

# plot of local maxima indices (accurate slice points) over EEG
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(time_y_20k, y_20k, '-k')

for trigger_time in time_y_20k[local_maxima_indices]:
    ax.axvline(trigger_time, color='b', linewidth=1)

plt.xlabel("Seconds (s)", fontsize=10)
plt.ylabel('Voltage ($\mu$V)', fontsize=10)
plt.tight_layout()
plt.close()

maxima_diffs = np.diff(local_maxima_indices)


# for 30s, 20 volumes. For 90s, 60

def AAS_niaz(eeg, n_vols, n_slices_vol, slice_times, epochs):
    """input_var:
            eeg: sliced EEG data during MR sequence
            n_vols: number of volumes in the MR sequence
            n_slices_vol: number of slices per volume 
            slice_times: list of slice starting times
            epochs: number of epochs to be used in the avg artifact
    """
    clean_data = np.zeros_like(eeg)
    # For loop over n_vols:
    for i_vol in range(n_vols):
        print("i_vol: ", i_vol)
        tx = i_vol*n_slices_vol
        print("\ntx: ", tx)
        ty = tx+n_slices_vol
        print("\nty: ", ty)
        i_vol_slice_times = slice_times[tx:ty] # slice starts
        print(f"slice times for volume {i_vol+1}: ", i_vol_slice_times)
        
        print(f"Starting AAS for Volume {i_vol+1}")
        # For loop over n_slices (26) per volume
        for i_slice in range(n_slices_vol): 
            print("i_vol: ", i_vol)
            print("i_slice: ", i_slice)

            if i_vol == max(range(n_vols)) and i_slice >= (epochs-n_slices_vol):
                continue
            
            start_time = i_vol_slice_times[i_slice]
            print("start_time: ", start_time)
            
            start_time_idx = slice_times.index(start_time)
            print("start_time_idx: ", start_time_idx)
            

            starts_idxs = np.arange(start_time_idx, start_time_idx+n_slices_vol*epochs, n_slices_vol)
            
            slice_starts = []
            for idxs in starts_idxs:
                slice_time = slice_times[idxs]
                slice_starts.append(slice_time)
            print('slice_starts before checking %25/26==0: ', slice_starts)
            
            max_slice_idx = len(slice_times) 
            
            if slice_times.index(slice_starts[0]) >= max_slice_idx - epochs: # based off epochs. >= max_slice_idx - epochs
                continue
            else:
                pass
            
            slice_lengths = []

            for slice in slice_starts:

                # Do calc 
                start_idx = slice_times.index(slice) 
                print("start_idx: ", start_idx)
                end_idx = start_idx + 1 
                print("end_idx: ", end_idx)
                real_length = slice_times[end_idx] - slice_times[start_idx]
                print("real_length: ", real_length)
                # store into slice_lengths list 
                slice_lengths.append(real_length)
            
            print(f"\nStarting AAS for slice {i_slice+1}: ", slice_starts[0])
            print('\nslice_starts: ', slice_starts)
            #slice_dur = np.diff(slice_starts)    
            print("\nslice lengths:", slice_lengths)
            
            lmax = max(slice_lengths)
            print("\nmax slice duration:", lmax)
            
            avg_arr = np.nan*np.zeros((epochs, lmax))
            print("\narray of all slice blocks for artifact template creation:\n", avg_arr,
                  "\narray shape:", avg_arr.shape)
            
            # For loop to  create Avg Artifact 

            for i_epoch in range(epochs):
                epoch_length = slice_lengths[i_epoch]
                epoch_start = slice_starts[i_epoch]
                epoch = eeg[epoch_start:epoch_start+epoch_length]
                print(f"\nstarting time for block {i_epoch+1}:", epoch_start, 
                       "type: ", type(epoch), 
                       "shape: ", epoch.shape)
                for k, element in enumerate(epoch):
                    avg_arr[i_epoch, k] = element
            
            # Creating avg from avg_arr (ignoring nans)
            avg = np.nanmean(avg_arr, axis=0)
            
            avg_corr = avg[:slice_lengths[0]] # avg correct; sliced through to length
                
            # Avg artifact created, ready for subtraction from i_slice
            t_start = slice_starts[0]
            print('t_start: ', t_start)
            t_end = slice_starts[0]+slice_lengths[0]
            print('t_end: ', t_end)
            
            alpha = np.dot(eeg[t_start:t_end], avg_corr)/np.dot(avg_corr, avg_corr)
                
            clean_data[t_start:t_end] = eeg[t_start:t_end] - (alpha * avg_corr)    
            print("\nSuccessfully subtracted avg for start time: ", slice_starts[0])
  
    return clean_data


test1 = AAS_niaz(eeg=y_20k, 
                 n_vols=21, 
                 n_slices_vol=26, 
                 slice_times=local_maxima_indices,
                 epochs=10)

test1_30 = test1[:30*20000]

time_test1_30 = np.arange(len(test1_30)) / 20000


# plot of 30s EEG before and after AAS
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(time_y_20k_thirtyseg, y_20k_thirtyseg, '-k')
ax[0].set_title('Raw 30s Slice')

ax[1].plot(time_test1_30, test1_30, '-k')
ax[1].set_title('AAS 30s Slice')
plt.xlabel("Time (s)", fontsize=10)
plt.ylabel('Amplitude ($\mu$V)', fontsize=10)
plt.tight_layout()
plt.close()

# Filtering, down-sampling, ANC

# Creating artificial trigger channel 
# for 17 Hz 
slice_trigger_chan = np.zeros(len(y_20k)) 

for index in local_maxima_indices:
    try:
        slice_trigger_chan[index] = 1
    except:
        continue

time_slice_trigger_chan = np.arange(len(slice_trigger_chan)) / 20000

# plot of artificial trigger channel 
plt.figure()
plt.plot(time_slice_trigger_chan, slice_trigger_chan, '-k')
plt.close()

# Down-sampling artificial trigger channel
slice_trigger_chan_10kfs =  decimate(slice_trigger_chan, q=2, ftype='fir') 

slice_trigger_chan_1kfs = decimate(slice_trigger_chan_10kfs, q=10, ftype='fir')

# Down-sampling 30 s AAS EEG

test1_30_10k = decimate(test1_30, q=2,  ftype='fir') 
test1_30_1k = decimate(test1_30_10k, q=10, ftype='fir')

# Filtering 
# BPF 1-40
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


test1_30_1k_40lpf = bp_filter(test1_30_1k, f_lo=1, f_hi=40, fs=1000)
slice_trigger_chan_1kfs_1to40bpf = bp_filter(slice_trigger_chan_1kfs, f_lo=1, f_hi=40, fs=1000)


# Creating reference from artifical trigger channel, 30 s 
slice_ref_30 = slice_trigger_chan_1kfs_1to40bpf[10000:40000]
slice_ref_500x = slice_ref_30 * 500

plt.figure()
plt.plot(slice_ref_30, '-k')
plt.plot(slice_ref_500x, '-b')
plt.close()

slice_ref_500x = slice_ref_30 * 500

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

out, y = fastranc(slice_ref_500x,
                  test1_30_1k_40lpf,
                  N=60,
                  mu=0.001)  

out = out.flatten()

time_out = np.arange(len(out)) / 1000

plt.figure()
plt.plot(time_out, test1_30_1k_40lpf, '-k', alpha=0.5, label='Original' )
plt.plot(time_out, out, '-b', label='ANC')
plt.plot(time_out, slice_ref_500x, '-r', label='Ref')
plt.legend()
plt.show()

# psd to compare
freqs_og, psd_og = welch(test1_30_1k_40lpf, fs=1000, nperseg=1024)

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

