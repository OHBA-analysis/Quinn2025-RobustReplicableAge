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


def my_bootstrap_iter(design, data, config, info, contrast):
    I = resample(np.arange(data.num_observations), n_samples=data.num_observations, replace=True)
    tmp_design = deepcopy(design)
    tmp_design.design_matrix = tmp_design.design_matrix[I, :]

    tmp_data = deepcopy(data)
    tmp_data.data = tmp_data.data[I, ...]

    model = glm.fit.OLSModel(tmp_design, tmp_data)
    tmp_glm = PatchedGroupSensorGLMSpectrum(model, design, config, info,
                                            fl_contrast_names=[], data=tmp_data)
    cf2 = cohens_f2(tmp_glm, reg_idx=contrast)[0, 0, :, :]
    return cf2[None, ...]


#%% ----------------------------------------------------


parser = argparse.ArgumentParser(description="Process some inputs.")

# Text input with four options
parser.add_argument('--dataset', choices=['camcan', 'cambridge', 'oxford', 'nottingham'], 
                    default='camcan', required=False, help='Select one of the four options')

# Number of parallel processes
parser.add_argument('--nprocesses', type=int, required=False, default=1,
                    help='Number of parallel processes to use')

# Number of non-parametric permutations 
parser.add_argument('--nperms', type=int, required=False, default=2500,
                    help='Number of non-parametric permutations to compute for hypothesis test')

# Sensor normalisation
parser.add_argument('--ztrans', action=argparse.BooleanOptionalAction, default=True,
                    help='Whether to work with the normalised or unnormalised sensor data')

# Number of bootstraps
parser.add_argument('--nbootstraps', type=int, required=False, default=2500,
                    help='Number of bootstrap resamples to compute for effect sizes')

args = parser.parse_args()
log.info('Input options : {0}'.format(args))


#%% ----------------------------------------------------

gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level')
os.makedirs(gdir, exist_ok=True)
log.info('Saving results into : {}'.format(gdir))

outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])
outp = os.path.join(gdir, bigmeg_meta['data']['glm']['group_perm_base'])

ipicks = 'grad'
inorm = 'ztrans' if args.ztrans else 'noztrans'

gglmsp = []
cf2 = []

if args.dataset == 'camcan':
    ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_firstlevel')
    fname = bigmeg_meta['data']['glm']['camcan_firstlevel_base'] 
    df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'subj_data', 'all_collated_camcan.csv'))
    dataset = 'CamCAN'
elif args.dataset == 'cambridge':
    ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'cambridge_firstlevel')
    fname = bigmeg_meta['data']['glm']['cambridge_firstlevel_base'] 
    df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'subj_data', 'all_collated_camb.csv'))
    dataset = 'MEGUKCambridge'
elif args.dataset == 'oxford':
    ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'oxford_firstlevel')
    fname = bigmeg_meta['data']['glm']['oxford_firstlevel_base'] 
    df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'subj_data', 'all_collated_oxford.csv'))
    dataset = 'MEGUKOxford'
elif args.dataset == 'nottingham':
    ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'nottingham_firstlevel')
    fname = bigmeg_meta['data']['glm']['nottingham_firstlevel_base'] 
    df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'subj_data', 'all_collated_notts.csv'))
    dataset = 'MEGUKNottingham'
    ipicks = 'mag'  # only mags here

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

log.info('Fitting group model')
gDC = glm.design.DesignConfig()
gDC.add_regressor(name='Mean', rtype='Constant')
gDC.add_regressor(name='Age', rtype='Parametric', datainfo='Age', preproc='z')
gDC.add_simple_contrasts()

gglmsp = osl.glm.group_glm_spectrum(glm_fnames, design_config=gDC, datainfo={'Age': age})
gglmsp = PatchedGroupSensorGLMSpectrum(gglmsp.model, gglmsp.design, gglmsp.config, gglmsp.info,
                                        fl_contrast_names=gglmsp.fl_contrast_names, data=gglmsp.data)
outfile = outf.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm)
gglmsp.save_pkl(outfile, save_data=True)
log.info('Saved file : {0}'.format(outfile))

#%% ----------------------------------------------------
# Null Hypothesis Significance Test

log.info('Running null hypothesis significance test')

iperm = 1
nperms = args.nperms
P = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp, iperm, 0, nperms=nperms, nprocesses=args.nprocesses)

outfile = outp.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm, contrast='age')
P.save_pkl(outfile)
log.info('Saved file : {0}'.format(outfile))

#%% ----------------------------------------------------
# Effect Size

log.info('Computing effect size')

cf2 = cohens_f2(gglmsp, reg_idx=1)[0, 0, :, :]
outfile = outp.format(dataset=dataset, analysis='glmspectrum-cf2', model='age', sensor=ipicks, norm=inorm, contrast='age')
np.save(outfile, cf2)
log.info('Saved file : {0}'.format(outfile))

#%% ----------------------------------------------------
# Bootstrapped Effect Size

log.info('Bootstrapping effect size')

nstraps = args.nbootstraps
cf2 = np.zeros((nstraps, 102, 189))

# First strap from observed data
gglmsp = osl.glm.group_glm_spectrum(glm_fnames, design_config=gDC, datainfo={'Age': age})
gglmsp = PatchedGroupSensorGLMSpectrum(gglmsp.model, gglmsp.design, gglmsp.config, gglmsp.info,
                                        fl_contrast_names=gglmsp.fl_contrast_names, data=gglmsp.data)
cf2_observed = cohens_f2(gglmsp, reg_idx=1)[0, 0, None, :, :]

# Loop through remaining
strapfunc = partial(my_bootstrap_iter, gglmsp.design, gglmsp.data,
                    gglmsp.config, gglmsp.info, 1)
cf2 = Parallel(n_jobs=args.nprocesses, verbose=1)(delayed(strapfunc)() for _ in range(nstraps-1))
cf2.append(cf2_observed)
cf2 = np.concatenate(cf2, axis=0)

cf2pct = np.percentile(cf2, [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99], axis=0)

outfile = outp.format(dataset=dataset, analysis='glmspectrum-cf2-bootstrap', model='age', sensor=ipicks, norm=inorm, contrast='age')
np.save(outfile, cf2pct)
log.info('Saved file : {0}'.format(outfile))
