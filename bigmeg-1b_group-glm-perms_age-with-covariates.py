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
                          exclude_for_hippovol, exclude_for_leverage,
                          exclude_for_missing, get_matched_vect,
                          glm_effect_size_calculation, load_cardio,
                          load_headpos, load_headvol, load_matched_from_csv)

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

# Number of parallel processes
parser.add_argument('--nprocesses', type=int, required=False, default=1,
                    help='Number of parallel processes to use')

# Number of non-parametric permutations 
parser.add_argument('--nperms', type=int, required=False, default=2500,
                    help='Number of non-parametric permutations to compute for hypothesis test')

# Number of bootstraps
parser.add_argument('--nbootstraps', type=int, required=False, default=2500,
                    help='Number of bootstrap resamples to compute for effect sizes')

args = parser.parse_args()
log.info('Input options : {0}'.format(args))


#%% ----------------------------------------------------


log.info('Running GLMs with covariates on CamCAN')

gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level', 'agepluscov')
os.makedirs(gdir, exist_ok=True)

outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])
outp = os.path.join(gdir, bigmeg_meta['data']['glm']['group_perm_base'])
outc = os.path.join(gdir, bigmeg_meta['data']['glm']['group_cf2_base'])

ipicks = 'grad'
inorm = 'ztrans'

gglmsp = []
cf2 = []

log.info('Identifying first-level GLM files')
ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_firstlevel')
fname = bigmeg_meta['data']['glm']['camcan_firstlevel_base']
dataset = 'CamCAN'
st = osl.utils.Study(os.path.join(ddir, fname))

log.info('Loading and checking covariates')
df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'all_collated_camcan.csv'))
df = load_headvol(df)
df = load_cardio(df)

# Check that input files match meta data rows
fnames = st.get(task='resteyesclosed', sensor=ipicks, norm=inorm)
glm_fnames = []
for idx, ifname in enumerate(fnames):

    subind = 1
    subj = ifname.split('/')[-1].split('_')[subind].split('-')[1]

    row_match = np.where(df['ID'] == subj)[0]
    if len(row_match) > 0:
        glm_fnames.append(ifname)

subj_ids = [ff.split('/')[-1].split('_')[1].split('-')[1] for ff in glm_fnames]

covs = load_matched_from_csv(subj_ids, df)

covs = load_headpos(glm_fnames, covs)
glm_fnames, covs = exclude_for_missing(glm_fnames, covs)
glm_fnames, covs = exclude_for_hippovol(glm_fnames, covs)

outfile = os.path.join(gdir, 'group_glm_covariates.csv')
df2 = pd.DataFrame.from_dict(covs)
df2.to_csv(outfile)
log.info('Saved file : {0}'.format(outfile))

#%% ----------------------------------------------------


covs_to_check = ['age', 'GM_Vol_Norm', 'sex', 'Brain_Vol',
       'WM_Vol_Norm', 'Hippo_Vol_Norm', 'BPM', 'BPSys', 'BPDia',
       'Height', 'Weight', 'head_radius', 'x', 'y', 'z']

covs_titles = ['Age', 'Grey Matter\nVolume', 'Sex', 'Brain Volume',
       'White Matter\nVolume', 'Hippocampus\nVolume', 'Heart Rate (BPM)', 
       'Systolic\nBlood Pressure', 'Diastolic\nBlood Pressure',
       'Height', 'Weight', 'Head Radius', 
       'Head Position\n(Left-Right)', 'Head Position\n(Up-Down)', 
       'Head Position\n(Forward-Back)']


for ii in range(len(covs_to_check)):

    log.info('-'*25)
    log.info('Processing covariate {0}/{1} : {2}'.format(ii+1, len(covs_to_check), covs_to_check[ii]))

    # Don't compute multiple regression model for age, only simple linear
    if covs_to_check[ii] != 'age':

        #%% ----------------------------------------------------
        # Multiple regression model

        log.info('1. Fitting multiple regression model')
        gDC = glm.design.DesignConfig()
        gDC.add_regressor(name='Mean', rtype='Constant')
        gDC.add_regressor(name='Age', rtype='Parametric', datainfo='age', preproc='z')
        gDC.add_regressor(name=covs_titles[ii], rtype='Parametric', datainfo=covs_to_check[ii], preproc='z')
        gDC.add_simple_contrasts()

        cov_name = covs_to_check[ii].replace('_', '')

        gglmsp = osl.glm.group_glm_spectrum(glm_fnames, design_config=gDC, datainfo=covs)
        gglmsp = PatchedGroupSensorGLMSpectrum(gglmsp.model, gglmsp.design, gglmsp.config, gglmsp.info,
                        fl_contrast_names=gglmsp.fl_contrast_names, data=gglmsp.data)

        outfile = outf.format(dataset=dataset, analysis='glmspectrum', model='age-' + cov_name, sensor=ipicks, norm=inorm)
        gglmsp.save_pkl(outfile, save_data=True)
        log.info('Saved file : {0}'.format(outfile))

        #%% ----------------------------------------------------
        # Null Hypothesis Significance Test

        log.info('2. Running null hypothesis significance test for Age')

        P = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp, 1, 0,
                                            nperms=args.nperms,
                                            nprocesses=args.nprocesses)
        outfile = outp.format(dataset=dataset, 
                            analysis='glmspectrum', 
                            model='age-' + cov_name, 
                            sensor=ipicks, 
                            norm=inorm, 
                            contrast='age')
        P.save_pkl(outfile)
        log.info('Saved file : {0}'.format(outfile))

        log.info('3. Running null hypothesis significance test for {0}'.format(covs_to_check[ii]))
        P = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp, 2, 0,
                                            nperms=args.nperms,
                                            nprocesses=args.nprocesses)
        outfile = outp.format(dataset=dataset, 
                            analysis='glmspectrum', 
                            model='age-' + cov_name, 
                            sensor=ipicks, 
                            norm=inorm, 
                            contrast=cov_name)
        P.save_pkl(outfile)
        log.info('Saved file : {0}'.format(outfile))

        #%% ----------------------------------------------------
        # Effect Size

        log.info('4. Computing partial effect size of Age')
        cf2 = cohens_f2(gglmsp, reg_idx=1)[0, 0, :, :]
        outfile = outc.format(dataset=dataset, 
                            analysis='glmspectrum', 
                            model='age-' + cov_name, 
                            sensor=ipicks, 
                            norm=inorm, 
                            contrast='age')
        np.save(outfile, cf2)
        log.info('Saved file : {0}'.format(outfile))

        log.info('5. Computing partial effect size of {0}'.format(covs_to_check[ii]))
        cf2 = cohens_f2(gglmsp, reg_idx=2)[0, 0, :, :]
        outfile = outc.format(dataset=dataset, 
                            analysis='glmspectrum', 
                            model='age-' + cov_name, 
                            sensor=ipicks, 
                            norm=inorm, 
                            contrast=cov_name)
        np.save(outfile, cf2)
        log.info('Saved file : {0}'.format(outfile))

        #%% ----------------------------------------------------
        # Effect Size Bootstrap - GMV ONLY!

        if covs_to_check[ii] == 'GM_Vol_Norm':

            log.info('6. Bootstrapping partial effect size')
            nstraps = args.nbootstraps

            # Observed data
            cf2_observed = cohens_f2(gglmsp, reg_idx=1)[0, 0, None, :, :]

            # Loop through remaining
            strapfunc = partial(my_bootstrap_iter, gglmsp.design, gglmsp.data,
                                gglmsp.config, gglmsp.info, 1)
            cf2 = Parallel(n_jobs=args.nprocesses, verbose=1)(delayed(strapfunc)() for _ in range(nstraps-1))
            cf2.append(cf2_observed)
            cf2 = np.concatenate(cf2, axis=0)

            cf2pct = np.percentile(cf2, [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99], axis=0)

            outfile = outc.format(dataset=dataset, 
                                analysis='glmspectrum-cf2-bootstrap', 
                                model='age-' + cov_name, 
                                sensor=ipicks, 
                                norm=inorm, 
                                contrast='age')
            np.save(outfile, cf2pct)
            log.info('Saved file : {0}'.format(outfile))
        else:
            log.info('6. not bootstrapping this covariate')
 

    #%% ----------------------------------------------------
    # Simple linear model

    log.info('7. Fitting simple linear model')
    gDC = glm.design.DesignConfig()
    gDC.add_regressor(name='Mean', rtype='Constant')
    gDC.add_regressor(name=covs_titles[ii], rtype='Parametric', datainfo=covs_to_check[ii], preproc='z')
    gDC.add_simple_contrasts()

    cov_name = covs_to_check[ii].replace('_', '')

    gglmsp = osl.glm.group_glm_spectrum(glm_fnames, design_config=gDC, datainfo=covs)
    gglmsp = PatchedGroupSensorGLMSpectrum(gglmsp.model, gglmsp.design, gglmsp.config, gglmsp.info,
                      fl_contrast_names=gglmsp.fl_contrast_names, data=gglmsp.data)

    outfile = outf.format(dataset=dataset, analysis='glmspectrum', model='covonly-' + cov_name, sensor=ipicks, norm=inorm)
    gglmsp.save_pkl(outfile, save_data=True)
    log.info('Saved file : {0}'.format(outfile))

    #%% ----------------------------------------------------
    # Effect Size
    log.info('8. Computing effect size for simple linear covariate effect')
    cf2 = cohens_f2(gglmsp, reg_idx=1)[0, 0, :, :]
    outfile = outc.format(dataset=dataset, 
                          analysis='glmspectrum', 
                          model='covonly-' + cov_name, 
                          sensor=ipicks, 
                          norm=inorm, 
                          contrast=cov_name)
    np.save(outfile, cf2)
    log.info('Saved file : {0}'.format(outfile))

    #%% ----------------------------------------------------
    # Null Hypothesis Significance Test

    log.info('9. Running null hypothesis significance test for simple linear covariate effect')
    P = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp, 1, 0,
                                          nperms=args.nperms,
                                          nprocesses=args.nprocesses)
    outfile = outp.format(dataset=dataset, 
                          analysis='glmspectrum', 
                          model='covonly-' + cov_name, 
                          sensor=ipicks, 
                          norm=inorm, 
                          contrast=cov_name)
    P.save_pkl(outfile)
    log.info('Saved file : {0}'.format(outfile))
