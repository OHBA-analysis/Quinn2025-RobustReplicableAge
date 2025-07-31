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
                          glm_effect_size_calculation, subpanel_label,
                          load_matched_from_csv, load_headpos,
                          load_headvol, load_cardio, plot_sig_clusters_with_map2)

np.alltrue = np.all  # jfc

log = osl.logging.getLogger()
log.setLevel('INFO')

with open("pyproject_paths.toml", "rb") as f:
    bigmeg_meta = tomllib.load(f)


def load_age(fnames):
    df = pd.read_csv(os.path.join(bigmeg_meta['code_dir'], 'all_collated_camcan.csv'))
    df = load_headvol(df)
    df = load_cardio(df)

    glm_fnames = []
    age = []

    for idx, ifname in enumerate(fnames):
        #print('{0}/{1} - {2}'.format(idx, len(fnames), ifname.split('/')[-1]))

        #subind = 0 if args.dataset != 'camcan' else 1
        subind = 1
        subj = ifname.split('/')[-1].split('_')[subind].split('-')[1]

        row_match = np.where(df['ID'] == subj)[0]
        if len(row_match) > 0:
            glm_fnames.append(ifname)
            age.append(df.iloc[row_match]['Fixed_Age'].values[0])

    subj_ids = [ff.split('/')[-1].split('_')[1].split('-')[1] for ff in glm_fnames]

    covs = load_matched_from_csv(subj_ids, df)
    covs = load_headpos(glm_fnames, covs)

    return glm_fnames, covs, subj_ids


#%% ----------------------------------------------------
# Definitions


ipicks = 'grad'

gDC = glm.design.DesignConfig()
gDC.add_regressor(name='Mean', rtype='Constant')
gDC.add_regressor(name='Age', rtype='Parametric', datainfo='age', preproc='z')
gDC.add_simple_contrasts()

gDC2 = glm.design.DesignConfig()
gDC2.add_regressor(name='Mean', rtype='Constant')
gDC2.add_regressor(name='Age', rtype='Parametric', datainfo='age', preproc='z')
gDC2.add_regressor(name='XPos', rtype='Parametric', datainfo='x', preproc='z')
gDC2.add_regressor(name='YPos', rtype='Parametric', datainfo='y', preproc='z')
gDC2.add_regressor(name='ZPos', rtype='Parametric', datainfo='z', preproc='z')
gDC2.add_simple_contrasts()


#%% ----------------------------------------------------
# Load TRANSDEF


def run_model(ddir, fname, norm, gDC):

    data_dir = os.path.join(ddir, fname)
    log.info('Loading data from : {0}'.format(data_dir))
    st = osl.utils.Study(data_dir)

    fnames = st.get(task='resteyesclosed', sensor=ipicks, norm=norm)
    glm_fnames, covs, subj_id = load_age(fnames)

    gglmsp = osl.glm.group_glm_spectrum(glm_fnames, design_config=gDC, datainfo=covs)
    gglmsp = PatchedGroupSensorGLMSpectrum(gglmsp.model, gglmsp.design, gglmsp.config, gglmsp.info,
                                            fl_contrast_names=gglmsp.fl_contrast_names, data=gglmsp.data)
    return gglmsp


log.info('Running models with Maxfilter Trans Pos')
ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_firstlevel')
fname = 'mf2pt2_sub-{subj}_ses-rest_task-rest_{preproc}-glm-{analysis}_{sensor}-{norm}.pkl'
gglmsp_trans_ztrans = run_model(ddir, fname, 'ztrans', gDC)
gglmsp_trans_noztrans = run_model(ddir, fname, 'noztrans', gDC)
gglmsp_trans_ztrans_pos = run_model(ddir, fname, 'ztrans', gDC2)
gglmsp_trans_noztrans_pos = run_model(ddir, fname, 'noztrans', gDC2)

log.info('Running models with Maxfilter Movecomp')
ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_firstlevel_movecomp')
fname = 'mf2pt2_sub-{subj}_ses-rest_task-rest_{preproc}-glm-{analysis}_{sensor}-{norm}.pkl'
gglmsp_move_ztrans = run_model(ddir, fname, 'ztrans', gDC)
gglmsp_move_noztrans = run_model(ddir, fname, 'noztrans', gDC)
gglmsp_move_ztrans_pos = run_model(ddir, fname, 'ztrans', gDC2)
gglmsp_move_noztrans_pos = run_model(ddir, fname, 'noztrans', gDC2)

log.info('Running models with Plain Maxfilter')
ddir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_firstlevel_nomove')
fname = 'mf2pt2_sub-{subj}_ses-rest_task-rest_{preproc}-glm-{analysis}_{sensor}-{norm}.pkl'
gglmsp_max_ztrans = run_model(ddir, fname, 'ztrans', gDC)
gglmsp_max_noztrans = run_model(ddir, fname, 'noztrans', gDC)
gglmsp_max_ztrans_pos = run_model(ddir, fname, 'ztrans', gDC2)
gglmsp_max_noztrans_pos = run_model(ddir, fname, 'noztrans', gDC2)


#%% ----------------------------------------------------
# Null Hypothesis Significance Test


iperm = 1
nperms = 250
Pz = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp_trans_ztrans, iperm, 0, nperms=nperms, nprocesses=10)
Pn = osl.glm.MaxStatPermuteGLMSpectrum(gglmsp_trans_noztrans, iperm, 0, nperms=nperms, nprocesses=10)

cf2z = cohens_f2(gglmsp_trans_ztrans, reg_idx=1)[0, 0, :, :]
cf2n = cohens_f2(gglmsp_trans_noztrans, reg_idx=1)[0, 0, :, :]


fig, cstats = plot_sig_clusters_with_map2(gglmsp_trans_ztrans, Pz, 95, base=0.5)
fig.set_size_inches(6, 8)
for ii in range(len(cstats)):
    idx = -2 - ii
    fstr = '{} - {}Hz'.format(np.round(cstats.iloc[-ii-1]['Min Freq'], 1), 
                              np.round(cstats.iloc[-ii-1]['Max Freq'], 1))
    fig.get_children()[-2].get_children()[idx].set_title(fstr, fontsize=6, y=0.9)


fig, cstats = plot_sig_clusters_with_map2(gglmsp_trans_noztrans, Pn, 95, base=0.5)
fig.set_size_inches(6, 8)
for ii in range(len(cstats)):
    idx = -2 - ii
    fstr = '{} - {}Hz'.format(np.round(cstats.iloc[-ii-1]['Min Freq'], 1), 
                              np.round(cstats.iloc[-ii-1]['Max Freq'], 1))
    fig.get_children()[-2].get_children()[idx].set_title(fstr, fontsize=6, y=0.9)
subpanel_label(fig.get_children()[-2], 'C', title='Statistical t-spectrum for age effect\n\n', yf=1.7)
fig.get_children()[-1].set_text('')

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure-6a-abs.png')
fig.savefig(figpath, dpi=100, transparent=True)

nsteps_age = 4
cmfun = plt.cm.cividis
colors = [(0.2, 0.2, 0.2), (0.9, 0, 0.9)] # first color is black, last is red
cmfun = LinearSegmentedColormap.from_list("Custom", colors, N=20)
cm = cmfun(np.linspace(0, 1, nsteps_age))
age_spec, linelabels = gglmsp_trans_noztrans.model.project_range(1, nsteps=nsteps_age)
fx = osl.glm.prep_scaled_freq(0.5, gglmsp_trans_noztrans.f)


fig = plt.figure(figsize=(6, 8))
ax = plt.subplot(211)
for ii in range(nsteps_age):
    plt.plot(fx[0], 1e3*age_spec[ii, 0, :, :].mean(axis=0), lw=2, color=cm[ii, :])
    osl.glm.decorate_spectrum(plt.gca(), ylabel='Absolute Magnitude (a.u.)')
    plt.xticks(fx[2], fx[1])
plt.xlim(fx[0][0], fx[0][90])
l = plt.legend([str(int(l)) + '' for l in linelabels], frameon=False, fontsize=11, ncol=4,
            bbox_to_anchor=(0.5, 0.95), loc="center",  bbox_transform=ax.transAxes)
l.set_title('Age (Years)')
#ax.set_facecolor((1.0, 0.47, 0.42))
subpanel_label(ax, 'A', title='GLM modelled spectrum change across age')

"""
nsteps_hist = 24
axins = ax.inset_axes([0.65, 0.55, 0.35, 0.25])
n, bins, patches = axins.hist(gglmsp_trans_noztrans.data.info['age'], nsteps_hist, color='green')
cm = cmfun(np.linspace(0, 1, nsteps_hist))
for c, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm[c, :])
for tag in ['top', 'right']:
    axins.spines[tag].set_visible(False)
axins.set_xlabel('Age (Years)', fontsize=9)
axins.set_ylabel('Proportion of\nParticipants', fontsize=9)
axins.set_yticks([])
axins.set_xticks(np.arange(2, 10)*10, labels=np.arange(2, 10)*10, fontsize=9)
axins.set_title('CamCAN (N={})'.format(gglmsp_trans_noztrans.data.num_observations))
"""

freqs = (5, 8, 10.5, 15, 36, 60)
ax = plt.subplot(212)
osl.glm.plot_joint_spectrum(gglmsp_trans_noztrans.f, cf2n.T, 
                            gglmsp_trans_noztrans.info, 
                            ax=ax, freqs=freqs, 
                            base=0.5, ylabel="Cohen's $F^2$")
subpanel_label(ax, 'B', title='Effect size spectrum for Age')

plt.subplots_adjust(hspace=0.35)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure-6b-abs.png')
fig.savefig(figpath, dpi=100, transparent=True)

#%% ----------------------------------------------------
# With headpos


freqs = (5, 8, 10.5, 15)
pargs = {'freqs': freqs, 'base': 0.5, 'metric': 'copes', 'title': ''}

plt.figure()
ax = plt.subplot(2, 3, 1)
gglmsp_move_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'A', title="SSS + Movecomp + trans")
#ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(2, 3, 4)
gglmsp_move_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'D', title="SSS Movecomp + trans + ztransform")
#ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 3, 2)
gglmsp_move_noztrans_pos.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'B', title="SSS Movecomp + trans + pos")
#ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(2, 3, 5)
gglmsp_move_ztrans_pos.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'E', title="SSS Movecomp + trans + pos + ztransform")
#ax.get_children()[11].set_ylim(-14, 14)

plt.subplots_adjust(hspace=0.4, wspace=0.4)




freqs = (5, 8, 10.5, 15)
pargs = {'freqs': freqs, 'base': 0.5, 'metric': 'tstats', 'title': ''}

fig = plt.figure(figsize=(16, 9))
ax = plt.subplot(2, 3, 1)
gglmsp_max_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'A', title="SSS")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 3, 4)
gglmsp_max_noztrans_pos.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'D', title="SSS + pos")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 3, 2)
gglmsp_move_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'B', title="SSS + Movecomp")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 3, 5)
gglmsp_move_noztrans_pos.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'E', title="SSS + Movecomp + pos")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 3, 3)
gglmsp_trans_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'C', title="SSS + Movecomp + trans")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 3, 6)
gglmsp_trans_noztrans_pos.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'F', title="SSS Movecomp + trans + pos")
ax.get_children()[11].set_ylim(-14, 14)

plt.subplots_adjust(hspace=0.4, wspace=0.4)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure-rik.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))





#%% ----------------------------------------------------
# SSS variants plot

freqs = (5, 8, 10.5, 15)
pargs = {'freqs': freqs, 'base': 0.5, 'metric': 'tstats', 'title': ''}

fig = plt.figure(figsize=(16, 9))
ax = plt.subplot(2, 4, 1)
gglmsp_max_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'A', title="Absolute Magnitude\nSSS")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 4, 5)
gglmsp_max_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'C', title="Relative Magnitude\nSSS")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 4, 2)
gglmsp_move_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, '', title="SSS + Movecomp")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 4, 6)
gglmsp_move_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, '', title="SSS + Movecomp")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 4, 3)
gglmsp_trans_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, '', title="SSS + Movecomp + trans")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 4, 7)
gglmsp_trans_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, '', title="SSS + Movecomp + trans")
ax.get_children()[11].set_ylim(-14, 14)

ax = plt.subplot(2, 4, 4)
ts = gglmsp_trans_noztrans.model.tstats[1, 0, :, :]
plt.pcolormesh(gglmsp_trans_noztrans.f, gglmsp_trans_noztrans.f, np.corrcoef(ts.T),
               cmap='RdBu_r', vmin=-1, vmax=1)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Frequency (Hz)')
plt.xticks(np.arange(10)*10)
plt.yticks(np.arange(10)*10)
plt.colorbar(label='Correlation Coefficient')
plt.gca().set_aspect('equal')
subpanel_label(ax, 'B', title="Frequency generalisation\nof spatial topography", yf=1.4)

ax = plt.subplot(2, 4, 8)
ts = gglmsp_trans_ztrans.model.tstats[1, 0, :, :]
plt.pcolormesh(gglmsp_trans_noztrans.f, gglmsp_trans_noztrans.f, np.corrcoef(ts.T),
               cmap='RdBu_r', vmin=-1, vmax=1)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Frequency (Hz)')
plt.xticks(np.arange(10)*10)
plt.yticks(np.arange(10)*10)
plt.colorbar(label='Correlation Coefficient')
plt.gca().set_aspect('equal')
subpanel_label(ax, 'D', title="Frequency generalisation\nof spatial topography", yf=1.4)

plt.subplots_adjust(hspace=0.4, wspace=0.4)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure-rik.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))


#%% ----------------------------------------------------
# BIG SSS variants plot


fx = osl.glm.prep_scaled_freq(0.5, gglmsp_max_noztrans.f)
cmfun = plt.cm.cividis
colors = [(0.2, 0.2, 0.2), (0.9, 0, 0.9)] # first color is black, last is red
cmfun = LinearSegmentedColormap.from_list("Custom", colors, N=20)
nsteps_age = 3
cm = cmfun(np.linspace(0, 1, nsteps_age))
proj_chans = np.arange(102)

fig = plt.figure(figsize=(16, 20))

ax = plt.subplot(4, 4, 1)
gglmsp_max_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'A', title="Absolute Magnitude\nSSS")
ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(4, 4, 2)
age_spec, ll = gglmsp_max_noztrans.model.project_range(1, nsteps_age)
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec[ii, 0, proj_chans, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(fx[2], fx[1])


ax = plt.subplot(4, 4, 3)
gglmsp_max_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'B', title="Relative Magnitude\nSSS")
ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(4, 4, 4)
age_spec, ll = gglmsp_max_ztrans.model.project_range(1, nsteps_age)
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec[ii, 0, proj_chans, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(fx[2], fx[1])


ax = plt.subplot(4, 4, 5)
gglmsp_move_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'C', title="SSS + Movecomp")
ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(4, 4, 6)
age_spec, ll = gglmsp_move_noztrans.model.project_range(1, nsteps_age)
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec[ii, 0, proj_chans, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(fx[2], fx[1])

ax = plt.subplot(4, 4, 7)
gglmsp_move_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'D', title="SSS + Movecomp")
ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(4, 4, 8)
age_spec, ll = gglmsp_move_ztrans.model.project_range(1, nsteps_age)
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec[ii, 0, proj_chans, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(fx[2], fx[1])

ax = plt.subplot(4, 4, 9)
gglmsp_trans_noztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'E', title="SSS + Movecomp + trans")
ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(4, 4, 10)
age_spec, ll = gglmsp_trans_noztrans.model.project_range(1, nsteps_age)
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec[ii, 0, proj_chans, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(fx[2], fx[1])

ax = plt.subplot(4, 4, 11)
gglmsp_trans_ztrans.plot_joint_spectrum(1, ax=ax, **pargs)
subpanel_label(ax, 'F', title="SSS + Movecomp + trans")
ax.get_children()[11].set_ylim(-14, 14)
ax = plt.subplot(4, 4, 12)
age_spec, ll = gglmsp_trans_ztrans.model.project_range(1, nsteps_age)
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec[ii, 0, proj_chans, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(fx[2], fx[1])


ax = plt.subplot(4, 2, 7)
ts = gglmsp_trans_noztrans.model.tstats[1, 0, :, :]
plt.pcolormesh(gglmsp_trans_noztrans.f, gglmsp_trans_noztrans.f, np.corrcoef(ts.T),
               cmap='RdBu_r', vmin=-1, vmax=1)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Frequency (Hz)')
plt.xticks(np.arange(10)*10)
plt.yticks(np.arange(10)*10)
plt.colorbar(label='Correlation Coefficient')
plt.gca().set_aspect('equal')
subpanel_label(ax, 'G', title="Frequency generalisation\nof spatial topography", yf=1.1)

ax = plt.subplot(4, 2, 8)
ts = gglmsp_trans_ztrans.model.tstats[1, 0, :, :]
plt.pcolormesh(gglmsp_trans_noztrans.f, gglmsp_trans_noztrans.f, np.corrcoef(ts.T),
               cmap='RdBu_r', vmin=-1, vmax=1)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Frequency (Hz)')
plt.xticks(np.arange(10)*10)
plt.yticks(np.arange(10)*10)
plt.colorbar(label='Correlation Coefficient')
plt.gca().set_aspect('equal')
subpanel_label(ax, 'H', title="Frequency generalisation\nof spatial topography", yf=1.1)

plt.subplots_adjust(hspace=0.4, wspace=0.35, left=0.05, right=0.95)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure-rik2.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))



#%% ----------------------------------------------------
# sensor renorm


gglmsp = deepcopy(gglmsp_trans_ztrans)

X = gglmsp.data.data
norm = gglmsp_trans_ztrans.data.data[:, 0, :, :].sum(axis=(2))
X[:, 0, :, :] = X[:, 0, :, :] / norm[..., None]
gglmsp.data.data = X

gglmsp.model = glm.fit.OLSModel(gglmsp.design, gglmsp.data)

plt.figure()
ax = plt.subplot(121)
gglmsp_trans_ztrans.plot_joint_spectrum(1, ax=ax, base=0.5, metric='tstats')
ax = plt.subplot(122)
gglmsp.plot_joint_spectrum(1, ax=ax, base=0.5, metric='tstats')




#%% ----------------------------------------------------
# sensor renorm

ts1 = gglmsp_trans_noztrans.model.tstats[1, 0, :, :]
ts2 = gglmsp_trans_ztrans.model.tstats[1, 0, :, :]
plt.pcolormesh(gglmsp_trans_noztrans.f, gglmsp_trans_noztrans.f, np.corrcoef(ts1.T, ts2.T)[:189, 189:],
               cmap='RdBu_r', vmin=-1, vmax=1)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Frequency (Hz)')
plt.xticks(np.arange(10)*10)
plt.yticks(np.arange(10)*10)
plt.colorbar(label='Correlation Coefficient')