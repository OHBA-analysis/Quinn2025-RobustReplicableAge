import os
import tomllib
import subprocess 

import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
from scipy import signal
import pandas as pd
from labellines import labelLine, labelLines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from osl.glm.glm_spectrum import get_mne_sensor_cols

from bigmeg_utils import (cohens_f2, glm_effect_size_calculation,
                          glm_power_calculation, glm_sample_size_calculation,
                          plot_sig_clusters_with_map, scatter_density,
                          plot_sig_clusters_with_map2, subpanel_label)

log = osl.logging.getLogger()
log.setLevel('INFO')

with open("pyproject_paths.toml", "rb") as f:
    bigmeg_meta = tomllib.load(f)

#%% ------------------------------------------
# Set directories and load data

log.info('Loading group level GLM')

gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_grouplevel')
gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level')
outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])
outp = os.path.join(gdir, bigmeg_meta['data']['glm']['group_perm_base'])

ipicks = 'grad'
inorm = 'noztrans'
dataset = 'CamCAN'

infile = outf.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm)
gglmsp = osl.glm.read_glm_spectrum(infile)

infile = outp.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm, contrast='age')
P = osl.glm.read_glm_spectrum(infile)
thresh = 95

#outfile = outp.format(dataset=dataset, analysis='glmspectrum-cf2-bootstrap', model='age', sensor=ipicks, norm=inorm, contrast='age')
#cf2_bs = np.load(outfile + '.npy')
cf2 = cohens_f2(gglmsp, 1)[0, 0, :, :]

dataset = 'CamCAN (N={})'.format(gglmsp.data.data.shape[0])

#%% ------------------------------------------
# Run simple power analyses

log.info('Running power analyses')

fx = osl.glm.prep_scaled_freq(0.5, gglmsp.f)
nsteps_age = 4
age_spec, linelabels = gglmsp.model.project_range(1, nsteps=nsteps_age)
cf2 = cohens_f2(gglmsp, reg_idx=1)[0, 0, :, :]
obs_power = glm_power_calculation(cf2, 
                              gglmsp.model.num_regressors, 
                              gglmsp.model.num_observations,
                              0.05)
obs_sample_size = glm_sample_size_calculation(cf2, gglmsp.model.num_regressors, 0.05, 0.8, max_N=2000)

cf_line = np.linspace(0, 0.35)
observations = [10, 25, 50, 80, 150, 474]
power_contours = np.zeros((50, len(observations)))
for ii in range(len(observations)):
    power_contours[:, ii] = glm_power_calculation(cf_line, 
                                                  gglmsp.model.num_regressors, 
                                                  observations[ii],
                                                  0.05)

cf_line = np.linspace(0.01, 0.35)
rel_power = [0.2, 0.4, 0.6, 0.8, 0.99]
sample_contours = np.zeros((50, len(rel_power)))
for ii in range(len(rel_power)):
    sample_contours[:, ii] = glm_sample_size_calculation(cf_line, 
                                                         gglmsp.model.num_regressors, 
                                                         0.05, rel_power[ii], max_N=200000)


# Effect size that has 90% power for this sample

cf_line2 = np.linspace(0, 0.2, 500)
sample_effect = glm_power_calculation(cf_line2, 
                                      gglmsp.model.num_regressors, 
                                      gglmsp.model.num_observations, 0.05)
sample_90 = cf_line2[np.where(sample_effect>0.9)[0][0]]


#%% ------------------------------------------
# Left half of figure 1

log.info('Plotting figure 1 - Left')

fig, cstats = plot_sig_clusters_with_map2(gglmsp, P, thresh, 
                                          base=0.5, nclusters=6, 
                                          jitter_levels=6)
fig.set_size_inches(8, 8)
fig.get_children()[8].grid(True)
fig.get_children()[-1].set_text('')

# Go backward through cstats
for ii in range(len(cstats)):
    idx = -2 - ii
    fstr = '{} - {}Hz'.format(np.round(cstats.iloc[-ii-1]['Min Freq'], 1), 
                              np.round(cstats.iloc[-ii-1]['Max Freq'], 1))
    fig.get_children()[-2].get_children()[idx].set_title(fstr, fontsize=6, y=0.9)

subpanel_label(fig.get_children()[9], 'C', title='Spectrum of t-values for Age effect\n', yf=1.3)
subpanel_label(fig.get_children()[7], 'D', title='Position sorted t-values for Age effect', yf=1.2)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure6a.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))

#%% ------------------------------------------
# Right half of figure 1

log.info('Plotting figure 1 - Right')

cmfun = plt.cm.cividis
colors = [(0.2, 0.2, 0.2), (0.9, 0, 0.9)] # first color is black, last is red
cmfun = LinearSegmentedColormap.from_list("Custom", colors, N=20)

cm = cmfun(np.linspace(0, 1, nsteps_age))

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(211)
for ii in range(nsteps_age):
    plt.plot(fx[0], 1e3*age_spec[ii, 0, :, :].mean(axis=0), lw=2, color=cm[ii, :])
    osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
    plt.xticks(fx[2], fx[1])
l = plt.legend([str(int(l)) + '' for l in linelabels], frameon=False, fontsize=11, ncol=4,
            bbox_to_anchor=(0.5, 0.95), loc="center",  bbox_transform=ax.transAxes)
l.set_title('Age (Years)')
subpanel_label(ax, 'A', title='GLM modelled spectrum change across age')

nsteps_hist = 24
axins = ax.inset_axes([0.65, 0.55, 0.35, 0.25])
n, bins, patches = axins.hist(gglmsp.data.info['Age'], nsteps_hist, color='green')
cm = cmfun(np.linspace(0, 1, nsteps_hist))
for c, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm[c, :])
for tag in ['top', 'right']:
    axins.spines[tag].set_visible(False)
axins.set_xlabel('Age (Years)', fontsize=9)
axins.set_ylabel('Proportion of\nParticipants', fontsize=9)
axins.set_yticks([])
axins.set_xticks(np.arange(2, 10)*10, labels=np.arange(2, 10)*10, fontsize=9)
axins.set_title('CamCAN (N={})'.format(gglmsp.data.num_observations))

freqs = (5, 8, 10.5, 15, 36, 60)
ax = plt.subplot(212)
osl.glm.plot_joint_spectrum(gglmsp.f, cf2.T, 
                            gglmsp.info, 
                            ax=ax, freqs=freqs, 
                            base=0.5, ylabel="Cohen's $F^2$")
subpanel_label(ax, 'B', title='Effect size spectrum for Age')

plt.subplots_adjust(hspace=0.35)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure6b.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))


#%% ------------------------------------------------
# Combined figure 6

figpatha = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure6a.png')
figpathb = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure6b.png')
figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure6.png')

cmd = f"montage -geometry +1+1 {figpathb} {figpatha} {figpath}"
subprocess.run(cmd.split(' '))
