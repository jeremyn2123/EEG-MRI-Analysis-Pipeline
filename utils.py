# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:10:11 2023

@author: z5265396
"""

import mne 
import numpy as np
import matplotlib.pyplot as plt 

def edf_layout():
    tenfive = mne.channels.read_layout("EEG1005")
    edf_chans=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
               'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz',
               'AF3', 'AF4', 'FC3', 'FC4', 'FT9', 'FT10', 'TP9', 'TP10',
               'CP5', 'CP6']
    n_chans = len(edf_chans)
    pos_edf = np.zeros((n_chans,4))
    ids_edf = [] 
    for i, ch in enumerate(edf_chans):
        ids_edf.append(i)
        if ch in tenfive.names:
            j = tenfive.names.index(ch)
            pos_edf[i,:] = tenfive.pos[j,:]
        else:
            print(ch, "NOT found...")
    return pos_edf, edf_chans, ids_edf


def quikcap_32_edf(filename):
    raw=mne.io.read_raw_edf(filename, preload=True)
    pos_edf, edf_chans, ids_edf = edf_layout()
    edf_lay = mne.channels.Layout(box=None, pos=pos_edf, names=edf_chans,
                                  ids=ids_edf, kind='eeg')
    return raw, edf_lay
    

def test_edf_layout():
    lay = edf_layout()
    print(lay)


ch_pos = {
    'Fp1': np.array([ -0.029    , 0.104536 ,  0.0331255]),
    'Fp2': np.array([0.03     , 0.104651 ,  0.0341187]),
    'F11': np.array([ -0.074    , 0.046771 , -0.0306181]),
    'F7': np.array([ -0.07     , 0.0618921,  0.0300395]),
    'F3': np.array([ -0.052    , 0.0767816,  0.0887107]),
    'Fz': np.array([0.002    , 0.0887299,  0.113494 ]),
    'F4': np.array([0.053    , 0.0786525,  0.0874859]),
    'F8': np.array([0.07     , 0.0612459,  0.0331351]),
    'F12': np.array([0.074    , 0.0444855, -0.0243113]),
    'FT11': np.array([ -0.08     , 0.0185638, -0.039413 ]),
    'FC3': np.array([ -0.062   , 0.04761 ,  0.106204]),
    'FCz': np.array([0.003    , 0.0572728,  0.137295 ]),
    'FC4': np.array([0.063    , 0.0483719,  0.104102 ]),
    'FT12': np.array([0.079    , 0.016394 , -0.0321129]),
    'T7': np.array([ -0.084     , 0.00351988,  0.0388537 ]),
    'C3': np.array([ -0.069    , 0.0147642,  0.118085 ]),
    'Cz': np.array([0.003    , 0.0220258,  0.154489 ]),
    'C4': np.array([0.069    , 0.0157575,  0.117969 ]),
    'T8': np.array([0.084     , 0.00398277,  0.0428268 ]),
    'CP3': np.array([-0.067    , -0.0229805, 0.122482 ]),
    'CPz': np.array([0.003    ,  -0.0178212,  0.158125 ]),
    'CP4': np.array([0.067    ,  -0.0190074,  0.122019 ]),
    'M1': np.array([-0.076     , -0.0412549 , 0.00883346]),
    'M2': np.array([0.077    ,  -0.0384583,  0.015555 ]),
    'P7': np.array([-0.073    , -0.0545051, 0.0506477]),
    'P3': np.array([-0.056    , -0.0551609, 0.11415  ]),
    'Pz': np.array([0.002    ,  -0.0529139,  0.142078 ]),
    'P4': np.array([0.056    ,  -0.0531743,  0.113919 ]),
    'P8': np.array([0.072    ,  -0.0521714,  0.0533961]),
    'O1': np.array([-0.029    , -0.0918544, 0.0670803]),
    'Oz': np.array([0.001    ,  -0.0940242,  0.0743804]),
    'O2': np.array([0.03     ,  -0.0907454,  0.0679578])
}

nasion = np.array([0.        , 0.11943411, 0.        ])
lpa = np.array([-0.11943411,  0.        ,  0.        ])
rpa= np.array([0.11943411, 0.        , 0.        ])

quikcap_32 = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                           nasion=nasion,
                                           lpa=lpa,
                                           rpa=rpa,
                                           coord_frame='head')

def fix_chans(raw):
    """Marks EKG and EMG as bad. Assigns VEOG, HEOG to EOG channel.
    Drops EKG and EMG (required for quikcap_32 montage to be applied)"""
    raw.info["bads"].extend(['EKG', 'EMG'])
    veog_idx = raw.ch_names.index('VEOG')
    heog_idx = raw.ch_names.index('HEOG')
    raw.set_channel_types({raw.ch_names[veog_idx]: 'eog', 
                       raw.ch_names[heog_idx]: 'eog'})
    raw.drop_channels(['EKG', 'EMG'])
    raw.set_montage(quikcap_32)

def main():
    print("main() has been called from __name__ == __main__ below...")
    input("...")
    test_edf_layout()


if __name__ == "__main__":
    print("utils.py has been called as the main script")
    print("...proceeding to run main()")
    main()


def bp_filter(data, f_lo, f_hi, fs):
    """Digital band pass filter (6-th order Butterworth)

    Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data
    """
    from scipy.signal import butter, filtfilt
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  # Nyquist frequency
    b_lo = f_lo/f_ny  # normalized frequency [0..1]
    b_hi = f_hi/f_ny  # normalized frequency [0..1]
    # band-pass filter parameters
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt
