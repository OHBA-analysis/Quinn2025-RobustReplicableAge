import argparse
import os
import pickle
import sys
import tomllib
from copy import deepcopy
from functools import partial

import glmtools
import glmtools as glm
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import signal, spatial, stats
from sklearn.utils import resample

from bigmeg_utils import (PatchedGroupSensorGLMSpectrum, cohens_f2,
                          glm_effect_size_calculation)

np.alltrue = np.all  # jfc

log = osl.logging.getLogger()
log.setLevel('INFO')

with open("pyproject_paths.toml", "rb") as f:
    bigmeg_meta = tomllib.load(f)


#%% ----------------------------------------------------


parser = argparse.ArgumentParser(description="Process some inputs.")

# Text input with four options
parser.add_argument('--dataset', choices=['camcan', 'cambridge', 'oxford', 'nottingham'], 
                    default='camcan', required=False, help='Select one of the four options')

args = parser.parse_args()
log.info('Input options : {0}'.format(args))


#%% ----------------------------------------------------

gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level')
os.makedirs(gdir, exist_ok=True)
log.info('Saving results into : {}'.format(gdir))

outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])
outp = os.path.join(gdir, bigmeg_meta['data']['glm']['group_perm_base'])

ipicks = 'grad'
inorm = 'ztrans'

gglmsp = []
cf2 = []

ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_firstlevel')
fname = bigmeg_meta['data']['glm']['camcan_firstlevel_base'] 
df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'all_collated_camcan.csv'))
dataset = 'CamCAN'

log.info('Running GLMs for {0}'.format(dataset))


#%% ----------------------------------------------------


log.info('Identifying first-level GLM files')

data_dir = os.path.join(ddir, fname)
log.info('Loading data from : {0}'.format(data_dir))
st = osl.utils.Study(data_dir)

subj_id = []
age = []
glm_fnames = []

group_cov = []

log.info('Loading age covariate')
fnames = st.get(task='resteyesclosed', sensor=ipicks, norm=inorm)
for idx, ifname in enumerate(fnames):
    #print('{0}/{1} - {2}'.format(idx, len(fnames), ifname.split('/')[-1]))

    subind = 0 if args.dataset != 'camcan' else 1
    subj = ifname.split('/')[-1].split('_')[subind].split('-')[1]

    row_match = np.where(df['ID'] == subj)[0]
    if len(row_match) > 0:
        glm_fnames.append(ifname)
        age.append(df.iloc[row_match]['Fixed_Age'].values[0])
        # ------- WORKSHOP
        group_cov.append({'age': df.iloc[row_match]['Fixed_Age'].values[0],
                     'brain_volume': df.iloc[row_match]['Brain_Vol'].values[0],
                     'grey_matter_volume': df.iloc[row_match]['GM_Vol_Norm'].values[0],
                     'white_matter_volume': df.iloc[row_match]['WM_Vol_Norm'].values[0],
                    })
df = pd.DataFrame.from_dict(group_cov).to_csv('camcan_group_covs.csv')


#%% -------------------------------
info = []

for ii in range(len(glm_fnames)):
    print(ii)
    gglmsp = osl.glm.read_glm_spectrum(glm_fnames[ii])
    subj_id = glm_fnames[ii].split('/')[-1][11:19]

    outbase = '/rds/projects/q/quinna-camcan/camcan_bigglm/processed-data/CamCAN_firstlevel/'
    fifname = outbase + f'mf2pt2_sub-{subj_id}_ses-rest_task-rest_megtransdef/mf2pt2_sub-{subj_id}_ses-rest_task-rest_megtransdef_preproc-raw.fif'
    icaname = outbase + f'mf2pt2_sub-{subj_id}_ses-rest_task-rest_megtransdef/mf2pt2_sub-{subj_id}_ses-rest_task-rest_megtransdef_ica.fif'

    raw = mne.io.read_raw(fifname, preload=True)
    ica = mne.preprocessing.read_ica(icaname)

    rejected_samples = np.isnan(raw.get_data(picks='meg', reject_by_annotation='NaN')).sum(axis=1).mean()
    rejected_pc = rejected_samples / raw.n_times  * 100

    eog_reject = len(ica.labels_['eog'])
    ecg_reject = len(ica.labels_['ecg'])
    tmp = ica.get_explained_variance_ratio(raw)
    ica_explained_mag = tmp['mag']
    ica_explained_grad = tmp['grad']

    vif_max = glm.design.variance_inflation_factor(gglmsp.design.design_matrix).max()
    singular_values = np.linalg.svd(gglmsp.design.design_matrix)[1]
    singular_values = singular_values / singular_values.max()
    sv_min = singular_values.min()

    subj = {'rejected_samples': rejected_samples,
            'rejected_pc': rejected_pc,
            'ica_explained_mag': ica_explained_mag,
            'ica_explained_grad': ica_explained_grad,
            'eog_reject': eog_reject,
            'ecg_reject': ecg_reject,
            'vif_max': vif_max,
            'sv_min': sv_min}
    info.append(subj)

df = pd.DataFrame.from_dict(info)



msg = "An average of {0}\% of data were marked as 'bad' (standard deviation: {1}, min: {2}, max: {3}) equivalent to an average of {4} seconds."
print(msg.format(df.rejected_pc.mean(),
                 df.rejected_pc.std(),
                 df.rejected_pc.min(),
                 df.rejected_pc.max(),
                 df.rejected_samples.mean()
                 ))

msg = "This decomposition explained an average of {0}\% ({1}\% gradiometers, {2}\% magnetometers) of variance in the sensor data across datasets."
print(msg.format(df.ica_explained.mean(),


msg = "Between {0} and {1} EOG components were rejected in each dataset, with an average of {2} (standard deviation: {3}) across all datasets."
print(msg.format(df.eog_reject.min(),
                 df.eog_reject.max(),
                 df.eog_reject.mean(),
                 df.eog_reject.std()
                 ))

msg = "Between {0} and {1} ECG components were rejected in each dataset, with an average of {2} (standard deviation: {3}) across all datasets."
print(msg.format(df.ecg_reject.min(),
                 df.ecg_reject.max(),
                 df.ecg_reject.mean(),
                 df.ecg_reject.std()
                 ))

msg = "The minimum singular value of the first level GLM design matrices had an average values of {0} with standard deviation of {1}"
print(msg.format(df.sv_min.mean(),
                 df.sv_min.std(),
                 ))

msg = "The maximum variance inflation factor of the first level GLM design matrices had an average values of {0} with standard deviation of {1}"
print(msg.format(df.vif_max.mean(),
                 df.vif_max.std(),
                 ))