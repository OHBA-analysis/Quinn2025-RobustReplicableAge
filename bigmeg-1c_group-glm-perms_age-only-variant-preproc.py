import os
import pickle
import sys
from copy import deepcopy

import glmtools
import glmtools as glm
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import signal, spatial, stats

from bigmeg_utils import (PatchedGroupSensorGLMSpectrum,
                          cohens_f2, glm_effect_size_calculation)

np.alltrue = np.all  # jfc

log = osl.logging.getLogger()
log.setLevel('INFO')

#%% ----------------------------------------------------

gdir = '/rds/projects/q/quinna-camcan/camcan_bigglm/processed-data/CamCAN_grouplevel'
outf = gdir + '/bigmeg-{dataset}_glm-{analysis}-{model}_{sensor}-{norm}_group-level.pkl'
outp = gdir + '/bigmeg-{dataset}_glm-{analysis}-{model}_{sensor}-{norm}_group-level_perm-{contrast}.pkl'

ipicks = 'grad'
inorm = 'ztrans'

gglmsp = []
cf2 = []

for ii in range(2):

    if ii == 0:
        ddir = '/rds/projects/q/quinna-camcan/camcan_bigglm/processed-data/CamCAN_firstlevel_nocovs'
        fname = 'mf2pt2_sub-{subj}_ses-rest_task-rest_megtransdef-glm-{analysis}_{sensor}-{norm}.pkl'
        dataset = 'CamCAN_nocov'
        df = pd.read_csv('/rds/homes/q/quinna/code/glm_bigmeg/all_collated_camcan.csv')
    elif ii == 1:
        ddir = '/rds/projects/q/quinna-camcan/camcan_bigglm/processed-data/CamCAN_firstlevel_nomove'
        fname = 'mf2pt2_sub-{subj}_ses-rest_task-rest_megtransdef-glm-{analysis}_{sensor}-{norm}.pkl'
        dataset = 'CamCAN_nomove'
        df = pd.read_csv('/rds/homes/q/quinna/code/glm_bigmeg/all_collated_camcan.csv')

    st = osl.utils.Study(os.path.join(ddir, fname))

    subj_id = []
    age = []
    glm_fnames = []

    fnames = st.get(task='resteyesclosed', sensor=ipicks, norm=inorm)
    for idx, ifname in enumerate(fnames):
        #print('{0}/{1} - {2}'.format(idx, len(fnames), ifname.split('/')[-1]))

        subind = 0 if ii > 0 else 1
        subj = ifname.split('/')[-1].split('_')[subind].split('-')[1]

        row_match = np.where(df['ID'] == subj)[0]
        if len(row_match) > 0:
            glm_fnames.append(ifname)
            age.append(df.iloc[row_match]['Fixed_Age'].values[0])

    gDC = glm.design.DesignConfig()
    gDC.add_regressor(name='Mean', rtype='Constant')
    gDC.add_regressor(name='Age', rtype='Parametric', datainfo='Age', preproc='z')
    gDC.add_simple_contrasts()
    gglmsp = osl.glm.group_glm_spectrum(glm_fnames, design_config=gDC, datainfo={'Age': age})
    gglmsp = blahblah(gglmsp.model, gglmsp.design, gglmsp.config, gglmsp.info,
                      fl_contrast_names=gglmsp.fl_contrast_names, data=gglmsp.data)


    outfile = outf.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm)
    gglmsp.save_pkl(outfile, save_data=True)

    cf2 = cohens_f2(gglmsp, reg_idx=1)[0, 0, :, :]
    outfile = outp.format(dataset=dataset, analysis='glmspectrum-cf2', model='age', sensor=ipicks, norm=inorm, contrast='age')
    np.save(outfile, cf2)

    iperm = 1
    nprocesses = 12
    nperms = 2500
    P = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp, iperm, 0, nperms=nperms, nprocesses=nprocesses)

    outfile = outp.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm, contrast='age')
    P.save_pkl(outfile)