import glob
import os
import tomllib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import pandas as pd
import glmtools as glm
from labellines import labelLine, labelLines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
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
# Set data directories

ipicks = 'grad'
inorm = 'ztrans'
dataset = 'CamCAN'

# Load core age model
gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level')
outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])

infile = outf.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm)
gglmsp = osl.glm.read_glm_spectrum(infile)

gdir2 = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level', 'agepluscov')
covfile = os.path.join(gdir2, 'group_glm_covariates.csv')
cov_df = pd.read_csv(covfile)

# Indentify covariate files
gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level', 'agepluscov')
outf = os.path.join(gdir, 'bigmeg-CamCAN_glm-glmspectrum-covonly-{contrast}_grad-ztrans_group-level.pkl')
outc = os.path.join(gdir, 'bigmeg-CamCAN_glm-glmspectrum-covonly-{contrast}_grad-ztrans_group-level_cf2-{contrast}.npy')
glm_files = sorted(glob.glob(outf.replace('{contrast}', '*')))
cf2_files = sorted(glob.glob(outc.replace('{contrast}', '*')))

to_plot_order = ['age', 'sex',  # Demographics
                 'BPM', 'BPSys', 'BPDia',   # Heart
                 'x', 'y', 'z',  # Acquisition 
                 'BrainVol', 'GMVolNorm', 'WMVolNorm', 'HippoVolNorm',  # Brain anatomy
                 'headradius', 'Height', 'Weight',  # Physiology
                 ] 

to_plot_labels = ['Age', 'Sex',  # Demographics
                 'Heart Rate', 'Systolic Bloop Pressure', 'Diastolic Bloop Pressure',   # Heart
                 'X (right-left)', 'Y (up-down)', 'Z (forward-back)',  # Acquisition 
                 'Total Brain Volume', 'Global Grey Matter Volume', 'Global White Matter Volume', 'Hippocampal Volume',  # Brain anatomy
                 'Head Radius', 'Height', 'Weight',  # Physiology
                 ] 

base_cf2 = []
ageplus_cf2 = []
agechange_cf2 = []
age_corr = []
idx = np.where([g.find('-age.npy') > -1 for g in cf2_files])[0][0]
age_cf2 = np.load(cf2_files[idx]).reshape(-1)

to_corr_order = ['age', 'sex',  # Demographics
                 'BPM', 'BPSys', 'BPDia',   # Heart
                 'x', 'y', 'z',  # Acquisition 
                 'Brain_Vol', 'GM_Vol_Norm', 'WM_Vol_Norm', 'Hippo_Vol_Norm',  # Brain anatomy
                 'head_radius', 'Height', 'Weight',  # Physiology
                 ] 

for ii in range(len(glm_files)):

    outc = gdir + '/bigmeg-CamCAN_glm-glmspectrum-covonly-{0}_grad-ztrans_group-level_cf2-{1}.npy'
    base_cf2.append(np.load(outc.format(to_plot_order[ii], to_plot_order[ii])).reshape(-1))

    if to_plot_order[ii] == 'age':
        continue

    outc = gdir + '/bigmeg-CamCAN_glm-glmspectrum-age-{0}_grad-ztrans_group-level_cf2-{1}.npy'
    ageplus_cf2.append(np.load(outc.format(to_plot_order[ii], to_plot_order[ii])).reshape(-1))

    outc = gdir + '/bigmeg-CamCAN_glm-glmspectrum-age-{0}_grad-ztrans_group-level_cf2-age.npy'
    agechange_cf2.append(np.load(outc.format(to_plot_order[ii])).reshape(-1) - age_cf2)
    #agechange_cf2.append(age_cf2 - np.load(outc.format(to_plot_order[ii])).reshape(-1))

    age_corr.append(np.corrcoef(cov_df['age'], cov_df[to_corr_order[ii]])[1, 0])


#%% ------------------------------------------
# Make covariate correlations summary figure

def quick_boxplot(data, positions, c):
    from scipy.stats import gaussian_kde

    plt.boxplot(data, positions=positions, notch=True, patch_artist=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c, marker='.'),
            medianprops=dict(color=c),
            )

    for idx, dat in enumerate(data):
        density = gaussian_kde(dat)
        xs = np.linspace(np.min(dat), np.max(dat), 200)
        kde = density(xs)
        kde = kde / np.max(kde) / 4 + positions[idx] + 0.3
        #plt.plot(kde, xs)
        plt.fill_betweenx(xs, np.ones_like(xs)*np.min(kde), kde, facecolor=c)


cols = plt.cm.Set1(np.arange(5))

fig = plt.figure(figsize=(9, 12))
plt.subplots_adjust(bottom=0.2)
plt.subplot(311)
quick_boxplot(base_cf2[:2], np.arange(len(base_cf2))[:2], cols[0, :])
quick_boxplot(base_cf2[2:5], np.arange(len(base_cf2))[2:5], cols[1, :])
quick_boxplot(base_cf2[5:8], np.arange(len(base_cf2))[5:8], cols[2, :])
quick_boxplot(base_cf2[8:12], np.arange(len(base_cf2))[8:12], cols[3, :])
quick_boxplot(base_cf2[12:], np.arange(len(base_cf2))[12:], cols[4, :])
plt.xticks(np.arange(len(base_cf2)), [])
#plt.xticks(np.arange(len(base_cf2)), to_plot_order, rotation=45, ha='left')
plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.xlim(-1, 15)
plt.ylabel("Cohen's $F^2$")
plt.grid(True)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
subpanel_label(plt.gca(), 'A', 'Effect size of covariate without age regressor')

plt.subplot(312)
quick_boxplot([agechange_cf2[1]], [1], cols[0, :])
quick_boxplot(agechange_cf2[1:4], np.arange(len(base_cf2))[2:5], cols[1, :])
quick_boxplot(agechange_cf2[4:7], np.arange(len(base_cf2))[5:8], cols[2, :])
quick_boxplot(agechange_cf2[7:11], np.arange(len(base_cf2))[8:12], cols[3, :])
quick_boxplot(agechange_cf2[11:], np.arange(len(base_cf2))[12:], cols[4, :])

plt.xticks(np.arange(len(base_cf2)), [])
plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.xlim(-1, 15)
plt.grid(True)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylabel("Change in Cohen's $F^2$")
subpanel_label(plt.gca(), 'B', 'Change in effect size of age with inclusion of covariate')

ax = plt.subplot(313)
plt.bar([1], age_corr[0], color=cols[0, :], )
plt.bar(np.arange(len(base_cf2))[2:5], age_corr[1:4], color=cols[1, :], )
plt.bar(np.arange(len(base_cf2))[5:8], age_corr[4:7], color=cols[2, :], )
plt.bar(np.arange(len(base_cf2))[8:12], age_corr[7:11], color=cols[3, :], )
plt.bar(np.arange(len(base_cf2))[12:], age_corr[11:], color=cols[4, :], )

#plt.xticks(np.arange(len(base_cf2)), [])
plt.xticks(np.arange(len(base_cf2)), to_plot_labels, rotation=45, ha='right')

plt.gca().tick_params(top=True, labeltop=False, bottom=False, labelbottom=True)
plt.xlim(-1, 15)
plt.grid(True)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylabel("Pearson's R")
subpanel_label(plt.gca(), 'C', title='Correlation between covariate and age')

leg_demo = mpatches.Patch(color=cols[0, :], label='Demographics')
leg_cardi = mpatches.Patch(color=cols[1, :], label='Cardiac')
leg_pos = mpatches.Patch(color=cols[2, :], label='Head Position')
leg_brain = mpatches.Patch(color=cols[3, :], label='Brain Anatomy')
leg_physio = mpatches.Patch(color=cols[4, :], label='Physiology')

plt.legend(handles=[leg_demo, leg_cardi, leg_pos, leg_brain, leg_physio], 
           frameon=False, fontsize=11, ncol=5,
           bbox_to_anchor=(0.5, -0.75), loc="center",  bbox_transform=ax.transAxes)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure7.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))


#%% ------------------------------------------
# Set directories and load data

log.info('Loading group level GLM')

gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'CamCAN_grouplevel')
gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level')
outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])
outp = os.path.join(gdir, bigmeg_meta['data']['glm']['group_perm_base'])

ipicks = 'grad'
inorm = 'ztrans'
dataset = 'CamCAN'

infile = outf.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm)
gglmsp1 = osl.glm.read_glm_spectrum(infile)

infile = outp.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm, contrast='age')
P1 = osl.glm.read_glm_spectrum(infile)

gdir = os.path.join(bigmeg_meta['output_dir'], 'processed-data', 'group_level', 'agepluscov')
outf = os.path.join(gdir, bigmeg_meta['data']['glm']['group_glm_base'])
outp = os.path.join(gdir, bigmeg_meta['data']['glm']['group_perm_base'])

infile = outf.format(dataset=dataset, analysis='glmspectrum', model='age-GMVolNorm', sensor=ipicks, norm=inorm)
gglmsp2 = osl.glm.read_glm_spectrum(infile)

infile = outp.format(dataset=dataset, analysis='glmspectrum', model='age-GMVolNorm', sensor=ipicks, norm=inorm, contrast='age')
P2 = osl.glm.read_glm_spectrum(infile)
thresh = 95




#%% ------------------------------------------
# Left half of figure 8

log.info('Plotting figure 8 - C')

fig, cstats = plot_sig_clusters_with_map2(gglmsp1, P1, thresh, base=0.5, nclusters=6)
fig.set_size_inches(8, 8)
fig.get_children()[8].grid(True)
yl = fig.get_children()[8].get_ylim()
fig.get_children()[-1].set_text('')

# Go backward through cstats
for ii in range(len(cstats)):
    idx = -2 - ii
    fstr = '{} - {}Hz'.format(np.round(cstats.iloc[-ii-1]['Min Freq'], 1), 
                              np.round(cstats.iloc[-ii-1]['Max Freq'], 1))
    fig.get_children()[-2].get_children()[idx].set_title(fstr, fontsize=6, y=0.9)

subpanel_label(fig.get_children()[9], 'C', title='Spectrum of t-values for Age effect\n', yf=1.3)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure8c.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))


#%% ------------------------------------------
# Left half of figure 8

log.info('Plotting figure 8 - D')

fig, cstats = plot_sig_clusters_with_map2(gglmsp2, P2, thresh, base=0.5, nclusters=6, yl=yl)
fig.set_size_inches(8, 8)
fig.get_children()[8].grid(True)
#fig.get_children()[8].set_ylim(-10, 10)
fig.get_children()[-1].set_text('')

# Go backward through cstats
for ii in range(len(cstats)):
    idx = -2 - ii
    fstr = '{} - {}Hz'.format(np.round(cstats.iloc[-ii-1]['Min Freq'], 1), 
                              np.round(cstats.iloc[-ii-1]['Max Freq'], 1))
    fig.get_children()[-2].get_children()[idx].set_title(fstr, fontsize=6, y=0.9)

subpanel_label(fig.get_children()[9], 'D', title='Spectrum of t-values for Age effect\ncontrolling for GGMV', yf=1.3)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure8d.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))