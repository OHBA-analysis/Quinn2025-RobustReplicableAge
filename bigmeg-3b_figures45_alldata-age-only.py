import os
import tomllib

import glmtools
import pandas as pd
import numpy as np
import glmtools as glm
import osl
import mne
import sys
from scipy import signal, stats, spatial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle 
from copy import deepcopy
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker

from bigmeg_utils import cohens_f2, glm_effect_size_calculation, plot_sig_clusters_with_map2, subpanel_label

np.alltrue = np.all  # jfc

log = osl.logging.getLogger()
log.setLevel('INFO')

with open("pyproject_paths.toml", "rb") as f:
    bigmeg_meta = tomllib.load(f)

#%% ----------------------------------------------------

ipicks = 'grad'
inorm = 'ztrans'


#%% ----------------------------------------------------
# Figure 5 - all sites in detail


cmfun = plt.cm.cividis
colors = [(0.2, 0.2, 0.2), (0.9, 0, 0.9)] # first color is black, last is red
cmfun = LinearSegmentedColormap.from_list("Custom", colors, N=20)

datasets = ['CamCAN (N={})',
            'MEG-UK Cambridge (N={})',
            'MEG-UK Oxford (N={})',
            'MEG-UK Nottingham (N={})']
dataset = ['CamCAN', 'MEGUKCambridge', 'MEGUKOxford', 'MEGUKNottingham']

gdir = '/rds/projects/q/quinna-camcan/camcan_bigglm/processed-data/CamCAN_grouplevel'
outf = gdir + '/bigmeg-{dataset}_glm-{analysis}-{model}_{sensor}-{norm}_group-level.pkl'
outp = gdir + '/bigmeg-{dataset}_glm-{analysis}-{model}_{sensor}-{norm}_group-level_perm-{contrast}.pkl'

ipicks = ['grad', 'grad', 'grad', 'mag']
inorm = 'ztrans'
thresh = 0.99
base = 0.5 

fig = plt.figure(figsize=(16, 12), constrained_layout=False)
gs = fig.add_gridspec(7, 4, hspace=1.5, wspace=0.6)
for ii in range(4):

    #%% ------------
    outfile = outf.format(dataset=dataset[ii], analysis='glmspectrum', model='age', sensor=ipicks[ii], norm=inorm)
    gglmsp = osl.glm.read_glm_spectrum(outfile)

    I = np.argsort(gglmsp.info._get_channel_positions()[:, 1])

    outfile = outp.format(dataset=dataset[ii], analysis='glmspectrum-cf2', model='age', sensor=ipicks[ii], norm=inorm, contrast='age')
    cf2 = np.load(outfile + '.npy')

    outfile = outp.format(dataset=dataset[ii], analysis='glmspectrum', model='age', sensor=ipicks[ii], norm=inorm, contrast='age')
    P = osl.glm.read_glm_spectrum(outfile)


    fx, xticklabels, xticks = osl.glm.prep_scaled_freq(base, gglmsp.f)

    #%% ------------
    obs = gglmsp.model.copes[1, 0, :, :]
    vm = np.abs(obs).max()

    ax0 = fig.add_subplot(gs[:2, ii])
    stitle = datasets[ii].format(gglmsp.data.data.shape[0])
    osl.glm.plot_sensor_spectrum(gglmsp.f, obs.T, gglmsp.info, ax=ax0, base=0.5, title=stitle + '\n')
    ax0.set_xlabel('')
    ax0.set_ylim(-0.00017, 0.00017)
    ax0.ticklabel_format(style='sci', scilimits=(-3,4),axis='y')

    axins = ax0.inset_axes([0.8, 0.9, 0.35, 0.25])
    key = 'Age'
    nsteps_hist = 24 if ii == 0 else 12
    n, bins, patches = axins.hist(gglmsp.data.info[key], nsteps_hist, color='green')
    cm = cmfun(np.linspace(0, 1, nsteps_hist))
    for c, p in enumerate(patches):
        plt.setp(p, 'facecolor', cm[c, :])
    for tag in ['top', 'right']:
        axins.spines[tag].set_visible(False)
    axins.set_xlabel('Age (Years)', fontsize=9)
    axins.set_ylabel('N PPTs', fontsize=9)
    axins.set_xticks(np.arange(2, 10, 2)*10, labels=np.arange(2, 10, 2)*10, fontsize=9)

    #%% ------------
    obs = gglmsp.model.get_tstats()[1, 0, :, :]
    vm = np.abs(obs).max()

    ax1 = fig.add_subplot(gs[2:4, ii])
    stitle = datasets[ii].format(gglmsp.data.data.shape[0])
    osl.glm.plot_sensor_spectrum(gglmsp.f, obs.T, gglmsp.info, ax=ax1, base=0.5, title='')
    ax1.set_xlabel('')

    #%% ------------
    ax2 = fig.add_subplot(gs[4, ii])
    X = np.r_[fx - np.diff(fx)[0]/2, fx[-1] + np.diff(fx)[-1]]
    ax2.pcolormesh(X, np.arange(obs.shape[0]+1), obs[I, :], vmin=-vm, vmax=vm, cmap='RdBu_r')
    ax2.pcolormesh(X, np.arange(obs.shape[0]+1), obs[I, :], vmin=-vm, vmax=vm, cmap='RdBu_r', alpha=0.4)

    ax2.set_xticks(xticks, xticklabels)

    #%% ------------
    ax3 = fig.add_subplot(gs[5:, ii])
    osl.glm.plot_sensor_spectrum(gglmsp.f, cf2.T, gglmsp.info, ax=ax3, base=0.5)
    ax3.set_ylim(0, 0.65)

    if ii == 3:
        axins2 = ax3.inset_axes([0.95, 0.6, 0.35, 0.25])
        osl.glm.plot_channel_layout(axins2, gglmsp.info)
        axins2.set_title('Sensor Positions')

    if ii == 0:
        ax3.set_ylabel("Cohen's $F^2$")
        ax0.set_ylabel('Parameter Estimate')
        ax1.set_ylabel('t-statistic')
        ax2.set_yticks([0, obs.shape[0]], ['Posterior', 'Anterior'])
        ax1.set_ylim(-15, 15)
        subpanel_label(ax0, 'A', title='Parameter Estimates\n\n\n', yf=1.3, xf=-0.5)
        subpanel_label(ax1, 'B', title='Null Hypothesis\nTest Statistic Spectrum', xf=-0.5)
        subpanel_label(ax2, 'C', title='Null Hypothesis\nTest Statistic 2D-Image', xf=-0.5)
        subpanel_label(ax3, 'D', title='Effect Sizes', xf=-0.5)

    else:
        ax2.set_yticks([0, obs.shape[0]], ['', ''])
        ax1.set_ylim(-8, 8)


figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure5.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))


#%% ---------------------------------------------
# Figure 4 - all sites comparison


datasets = ['CamCAN (N={})',
            'MEG-UK Cambridge (N={})',
            'MEG-UK Oxford (N={})',
            'MEG-UK Nottingham (N={})']
dataset = ['CamCAN', 'MEGUKCambridge', 'MEGUKOxford', 'MEGUKNottingham']

ipicks = ['grad', 'grad', 'grad', 'mag']
inorm = 'ztrans'
thresh = 0.99
base = 0.5 

# Load all models
gglmsp = list()
cf2 = []
for ii in range(4):
    outfile = outf.format(dataset=dataset[ii], analysis='glmspectrum', model='age', sensor=ipicks[ii], norm=inorm)
    gglmsp.append(osl.glm.read_glm_spectrum(outfile))

    outfile = outp.format(dataset=dataset[ii], analysis='glmspectrum-cf2', model='age', sensor=ipicks[ii], norm=inorm, contrast='age')
    cf2.append(np.load(outfile + '.npy'))

# Compute spatial correlations
C = np.zeros((gglmsp[0].model.tstats.shape[3], 4))
for ii in range(gglmsp[0].model.tstats.shape[3]):
    C[ii, 0] = np.corrcoef(gglmsp[0].model.tstats[1, 0, :, ii], gglmsp[1].model.tstats[1, 0, :, ii])[0, 1]
    C[ii, 1] = np.corrcoef(gglmsp[0].model.tstats[1, 0, :, ii], gglmsp[2].model.tstats[1, 0, :, ii])[0, 1]

    C[ii, 2] = np.corrcoef(cf2[0][:, ii], cf2[1][:, ii])[0, 1]
    C[ii, 3] = np.corrcoef(cf2[0][:, ii], cf2[2][:, ii])[0, 1]

# Compute nulls
nperms = 100
Cnull = np.zeros((gglmsp[0].model.tstats.shape[3], nperms))
for ii in range(nperms):
    I = np.random.permutation(np.arange(102))
    for jj in range(gglmsp[0].model.tstats.shape[3]):
        Cnull[jj, ii] = np.corrcoef(gglmsp[0].model.tstats[1, 0, :, jj], gglmsp[1].model.tstats[1, 0, I, jj])[0, 1]


fx, xticklabels, xticks = osl.glm.prep_scaled_freq(0.5, gglmsp[0].f)

fig = plt.figure(figsize=(16, 8), constrained_layout=False)
gs = fig.add_gridspec(3, 3)
plt.subplots_adjust(hspace=0.55, wspace=0.3)

for ii in range(3):
    ax = fig.add_subplot(gs[:2, ii])
    tmp_freq = []
    tmp_space = []
    for jj in range(4):

        if ii == 0:
            toplot_freq = gglmsp[jj].model.copes[1, 0, :, :].mean(axis=0)
            toplot_space = gglmsp[jj].model.copes[1, 0, :, :]
            tt = 'Age Parameter Estimates'
            yl = 'Contrast of Parameter Estiamtes'
        elif ii == 1:
            toplot_freq = gglmsp[jj].model.tstats[1, 0, :, :].mean(axis=0)
            toplot_space = gglmsp[jj].model.tstats[1, 0, :, :]
            tt = 'Age Null Hypothesis Test Statistics'
            yl = 't-statistics'
        else:
            toplot_freq = cf2[jj].mean(axis=0)
            toplot_space = cf2[jj]
            tt = 'Age Effect Sizes'
            yl = "Cohen's $F^2$"

        ll = datasets[jj].format(gglmsp[jj].data.data.shape[0])
        tmp_freq.append(toplot_freq[None, :])
        tmp_space.append(toplot_space[None, :])
        plt.plot(fx, toplot_freq, label=ll)
    subpanel_label(plt.gca(), chr(65+ii) + ': i', title=tt)
    plt.ylabel(yl)
    plt.xlabel('Frequency (Hz)')
    plt.xticks(xticks, xticklabels)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    if ii > 1:
        plt.legend(frameon=False, bbox_to_anchor=[0.65, 0.75])

    C = np.corrcoef(np.concatenate(tmp_freq, axis=0))
    axins = fig.add_subplot(gs[2, ii])
    Ci = np.eye(4)
    C[np.triu(C) != 0] = np.nan
    im = axins.imshow(C + Ci, vmin=-1, vmax=1, cmap='RdBu_r')
    for kk in range(4):
        for ll in range(4):
            if np.isnan(C[kk, ll]) == False:
                axins.text(ll, kk, str(np.round(C[kk, ll], 2)), 
                           ha='center', va='center', color='w',
                           fontsize=7)
    axins.set_aspect(1.0/axins.get_data_ratio(), adjustable='box')
    if ii > 1:
        plt.colorbar(im, ax=axins, label="Pearson's R")
    plt.xticks(np.arange(3), dataset[:3], rotation=25, ha='right')
    plt.yticks(np.arange(1, 4), dataset[1:])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    subpanel_label(plt.gca(), chr(65+ii) + ': ii', xf = -1, title='Spectrum Correlation')

    Cs = np.concatenate(tmp_space[:3], axis=0).reshape(3, -1)
    print(np.corrcoef(np.concatenate(tmp_space[:3], axis=0).reshape(3, -1)))


figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure4.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))
