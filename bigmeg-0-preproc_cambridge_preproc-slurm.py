import osl
import mne
import os
import sys
from scipy import stats
import sails
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% -------------------------------------------------------------

def preproc_zapline_dss(dataset, userargs, logfile=None):
    from meegkit import dss
    # https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407
    fline = userargs.get('fline', 50)
    nremove = userargs.get('nremove', 4)
    sfreq = dataset['raw'].info['sfreq'] # Extract the sampling freq

    for chtype in ['mag', 'grad']:
        inds = mne.pick_types(dataset['raw'].info, meg=chtype)
        data = dataset['raw'].get_data()[inds, :]

        #Apply MEEGkit toolbox function
        out, _ = dss.dss_line(data.T, fline, sfreq, nremove=nremove) # fline (Line noise freq) = 50 Hz for Europe

        dataset['raw']._data[inds, :] = out.T # Overwrite old data

    return dataset


def headsize_from_fids(fids):
    """Input is [3 x 3] array of [observations x dimensions]."""
    # https://en.wikipedia.org/wiki/Heron%27s_formula

    dists = spatial.distance.pdist(fids)
    semi_perimeter = np.sum(dists) / 2
    area = np.sqrt(semi_perimeter * np.prod(semi_perimeter - dists))

    return area

def get_device_fids(raw):
    # Put fiducials in device space
    head_fids = mne.viz._3d._fiducial_coords(raw.info['dig'])
    head_fids = np.vstack(([0, 0, 0], head_fids))
    fid_space = raw.info['dig'][0]['coord_frame']
    assert(fid_space == 4)  # Ensure we have FIFFV_COORD_HEAD coords

    # Get device to head transform and inverse
    dev2head = raw.info['dev_head_t']
    head2dev = mne.transforms.invert_transform(dev2head)
    assert(head2dev['from'] == 4)
    assert(head2dev['to'] == 1)

    # Apply transformation to get fids in device space
    device_fids = mne.transforms.apply_trans(head2dev, head_fids)

    return device_fids

def make_bads_regressor(raw, mode='raw'):
    bads = np.zeros((raw.n_times,))
    for an in raw.annotations:
        if an['description'].startswith('bad') and an['description'].endswith(mode):
            start = raw.time_as_index(an['onset'])[0] - raw.first_samp
            duration = int(an['duration'] * raw.info['sfreq'])
            bads[start:start+duration] = 1
    if mode == 'raw':
        bads[:int(raw.info['sfreq']*2)] = 1
        bads[-int(raw.info['sfreq']*2):] = 1
    else:
        bads[:int(raw.info['sfreq'])] = 1
        bads[-int(raw.info['sfreq']):] = 1
    return bads


def get_combined_grads_raw(raw):
    # This is probably done too early - 
    # literature suggests doing this immediately after computing the power spectra
    # which would be just before the GLM, not clear if we can average the GLM coefs
    try:
        from mne.viz.topomap import _merge_grad_data
    except ImportError:
        # This was moved at some point
        from mne.channels.layout import _merge_grad_data

    x = _merge_grad_data(raw.get_data(picks='grad'), method='mean')
    info = raw.copy().pick_types(meg='mag').info
    return mne.io.RawArray(x, info)


def get_sphere_fit_info(dataset, mode='mne'):

    if mode == 'maxfilter':
        fname = dataset['raw'].filenames[0].replace('megtransdef.fif', 'meg_sphere_fit.txt')
        spherefit = np.loadtxt(fname)
        head = spherefit[:3]
        radius = spherefit[3]
        notsure = spherefit[3:]
        return radius, head
    elif mode == 'mne':
        radius, head, device = mne.bem.fit_sphere_to_headshape(dataset['raw'].info)
        return radius, device


def load_cardio(subj_id):

    cardio = pd.read_excel('/rds/homes/q/quinna/code/glm_bigmeg/CardioMeasures_summary.xlsx')
    match_ind = np.where(cardio.CCID == subj_id)[0]

    bpm = cardio.iloc[match_ind[0]].pulse_mean if len(match_ind) == 1 else np.nan
    systol = cardio.iloc[match_ind[0]].bp_sys_mean if len(match_ind) == 1 else np.nan
    diastol = cardio.iloc[match_ind[0]].bp_dia_mean if len(match_ind) == 1 else np.nan
    height = cardio.iloc[match_ind[0]].height if len(match_ind) == 1 else np.nan
    weight = cardio.iloc[match_ind[0]].weight if len(match_ind) == 1 else np.nan

    cinfo = {'bpm': bpm, 'systol': systol, 'diastol': diastol, 'height': height, 'weight': weight}

    #log.info('Loaded cardio - {}'.format(cinfo))

    return cinfo


def run_first_level_new(dataset, userargs):
    run_id = osl.utils.find_run_id(dataset['raw'].filenames[0])
    subj_id = run_id.split('_')[1][4:]

    # Bad segments
    bads = make_bads_regressor(dataset['raw'], mode='mag')

    # EOGs - vertical only
    eogs = dataset['raw'].copy().pick_types(meg=False, eog=True)
    eogs = eogs.filter(l_freq=1, h_freq=20, picks='eog').get_data()

    # ECG - lots of sanity checking
    ecg_events, ch_ecg, av_pulse, = mne.preprocessing.find_ecg_events(dataset['raw'])
    ecg_events[:, 0] = ecg_events[:, 0] - dataset['raw'].first_samp
    ecg = np.zeros_like(bads)
    median_beat = np.median(np.diff(ecg_events[:, 0]))
    last_beat = median_beat
    for ii in range(ecg_events.shape[0]-1):
        beat = ecg_events[ii+1, 0] - ecg_events[ii, 0]
        if np.abs(beat-last_beat) > 50:
            beat = last_beat
        ecg[ecg_events[ii, 0]:ecg_events[ii+1, 0]] = beat
        beat = last_beat
    ecg[ecg==0] = median_beat
    ecg = ecg / dataset['raw'].info['sfreq'] * 60

    # Store covariates
    if userargs.get('covs', False):
        confs = {'VEOG': np.abs(eogs[0, :]), 'BadSegs': bads}
        covs = {'Linear': np.linspace(-1, 1, ecg.shape[0]), 'ECG': ecg}
    else:
        confs = None
        covs = None
    conds = None
    conts = None

    head_radius, head_pos = get_sphere_fit_info(dataset)
    dinfo = {'device_fids': get_device_fids(dataset['raw']),
             'avg_bpm': av_pulse,
             'head_radius': head_radius,
             'head_pos': head_pos}
    dinfo.update(load_cardio(subj_id))
    #log.info('Metadata : {}'.format(dinfo))

    #%% -------------------------------------------
    # GLM-Spectrum
    for picks in ['mag', 'grad']:
        if picks == 'grad':
            glm_raw = get_combined_grads_raw(dataset['raw'])
        else:
            glm_raw = dataset['raw'].copy().pick_types(meg=picks)
        fs = glm_raw.info['sfreq']

        #%% --------------------

        glmspec = osl.glm.glm_spectrum(glm_raw, fmin=1, fmax=95, 
                                    nperseg=int(fs * 2), noverlap=int(fs),
                                    mode='magnitude', axis=1,
                                    reg_ztrans=covs, reg_unitmax=confs,
                                    standardise_data=True)
        glmspec.model.extras = dinfo
            
        outname = '{subj_id}-glm-spectrum_{picks}-ztrans.pkl'.format(subj_id=run_id, picks=picks)
        pklname = os.path.join(userargs.get('outdir'), outname)
        glmspec.save_pkl(pklname)

        #%% --------------------

        glmspec = osl.glm.glm_spectrum(glm_raw, fmin=1, fmax=95, 
                                    nperseg=int(fs * 2), noverlap=int(fs),
                                    mode='magnitude',
                                    reg_ztrans=covs, reg_unitmax=confs,
                                    standardise_data=False)
        glmspec.model.extras = dinfo

        outname = '{subj_id}-glm-spectrum_{picks}-noztrans.pkl'.format(subj_id=run_id, picks=picks)
        pklname = os.path.join(userargs.get('outdir'), outname)
        glmspec.save_pkl(pklname)

    return dataset


#%% -------------------------------------------------------------

mask = pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)+pow(2,14)

# Identify bad channels prior to maxfilter
# Perhaps a few static bad channels

config = """
meta:
  event_codes:
    visual: 1
    auditory: 2
    button_press: 3
preproc:
  - crop:                 {tmin: 35}
  - find_events:          {min_duration: 0.005}
  - filter:               {l_freq: 0.25, h_freq: 150, method: 'iir', iir_params: {order: 5, ftype: butter}}
  - preproc_zapline_dss:  {'fline': 49, 'nremove': 4}
  - preproc_zapline_dss:  {'fline': 50, 'nremove': 8}
  - bad_segments:         {segment_len: 800, picks: 'mag'}
  - bad_segments:         {segment_len: 800, picks: 'grad'}
  - bad_segments:         {segment_len: 800, picks: 'mag', mode: diff}
  - bad_segments:         {segment_len: 800, picks: 'grad', mode: diff}
  - bad_channels:         {picks: 'mag'}
  - bad_channels:         {picks: 'grad'}
  - resample:             {sfreq: 250, n_jobs: 1}
  - ica_raw:              {picks: 'meg', n_components: 64}
  - ica_autoreject:       {picks: 'meg', ecgmethod: 'correlation'}
  - interpolate_bads:     {}
  - run_first_level_new:  {outdir: /rds/projects/q/quinna-camcan/camcan_bigglm/processed-data/CamCAN_firstlevel, covs: True}
"""
config = osl.preprocessing.load_config(config)
config['preproc'][-1]['run_first_level_new']['covs'] = False

#%% -------------------------------------------------------------

basedir = '/rds/projects/q/quinna-spectral-changes-in-ageing/mrc_meguk/raw_data/Cambridge/derivatives/'
oxfpath = os.path.join(basedir, 'sub-{subj}', 'meg' ,'sub-{subj}_task-{task}_proc-sss_meg.fif')
st = osl.utils.Study(oxfpath)

extrafuncs = [run_first_level_new]
inputs = st.get(task='resteyesopen') + st.get(task='resteyesclosed')

outdir = '/rds/projects/q/quinna-spectral-changes-in-ageing/mrc_meguk/processed_data/Cambridge_firstlevel/'

config['preproc'][-1]['run_first_level_new']['outdir'] = outdir

run_ind = int(sys.argv[1])
infiles = sorted(st.get())

np.alltrue = np.all  # stupid monkeypatch

dataset = osl.preprocessing.run_proc_batch(config, [infiles[run_ind]],
                                           outdir=outdir, overwrite=True, 
                                           extra_funcs=[run_first_level_new, preproc_zapline_dss])