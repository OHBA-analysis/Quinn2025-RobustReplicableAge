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
inorm = 'ztrans'
dataset = 'CamCAN'

infile = outf.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm)
gglmsp = osl.glm.read_glm_spectrum(infile)

infile = outp.format(dataset=dataset, analysis='glmspectrum', model='age', sensor=ipicks, norm=inorm, contrast='age')
P = osl.glm.read_glm_spectrum(infile)
thresh = 95

outfile = outp.format(dataset=dataset, analysis='glmspectrum-cf2-bootstrap', model='age', sensor=ipicks, norm=inorm, contrast='age')
cf2_bs = np.load(outfile + '.npy')

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

fig, cstats = plot_sig_clusters_with_map2(gglmsp, P, thresh, base=0.5, nclusters=6)
fig.set_size_inches(8, 8)
fig.get_children()[8].grid(True)
fig.get_children()[-1].set_text('')

# Go backward through cstats
for ii in range(len(cstats)):
    idx = -2 - ii
    fstr = '{} - {}Hz'.format(np.round(cstats.iloc[-ii-1]['Min Freq'], 1), 
                              np.round(cstats.iloc[-ii-1]['Max Freq'], 1))
    fig.get_children()[-2].get_children()[idx].set_title(fstr, fontsize=6, y=0.9)

subpanel_label(fig.get_children()[9], 'B', title='Spectrum of t-values for Age effect\n', yf=1.3)
subpanel_label(fig.get_children()[7], 'C', title='Position sorted t-values for Age effect', yf=1.2)

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure1a.png')
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
ax = plt.subplot(111)
for ii in range(nsteps_age):
    plt.plot(fx[0], 1e3*age_spec[ii, 0, :, :].mean(axis=0), lw=2, color=cm[ii, :])
    osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
    plt.xticks(fx[2], fx[1])
#plt.xlim(fx[0][0], fx[0][90])
l = plt.legend([str(int(l)) + '' for l in linelabels], frameon=False, fontsize=11, ncol=4,
            bbox_to_anchor=(0.5, 0.95), loc="center",  bbox_transform=ax.transAxes)
l.set_title('Age (Years)')
#ax.set_facecolor((1.0, 0.47, 0.42))
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

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure1b.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))

#%% ------------------------------------------------
# Combined figure 1

figpatha = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure1a.png')
figpathb = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure1b.png')
figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure1.png')

cmd = f"montage -geometry +1+1 {figpathb} {figpatha} {figpath}"
subprocess.run(cmd.split(' '))



#%% ------------------------------------------
# Define alpha peak helper functions
# Find alpha peak frequency and sensor for each dataset


def calc_parabolic(x, y):

    coefficients = np.polyfit(x, y, 2)

    # Calculate the vertex (peak) of the parabola
    a, b, c = coefficients
    x_vertex = -b / (2 * a)
    y_vertex = a * x_vertex**2 + b * x_vertex + c

    return x_vertex, y_vertex


def peak_find(f, pxx, low_f=5, hi_f=15):
    inds = np.logical_and(f > low_f, f < hi_f)
    base_f = [i for i, x in enumerate(inds) if x][0]
    alpha = pxx[inds]

    locs = signal.find_peaks(pxx[inds])[0]
    if len(locs) == 0:
        return (-1, -1)
    pks = alpha[locs]

    p = np.argsort(pks)[::-1]  # Sort from largest to smallest amp
    locs = locs[p]
    for ii in range(len(p)):
        if locs[ii] > 1 and locs[ii] < len(alpha) - 2:
            full_ind = base_f + locs[ii]
            pf, pm = calc_parabolic(f[full_ind-1:full_ind+2], pxx[full_ind-1:full_ind+2])
            return pf, pm
    return (-1, -1)


info = []

spectra = gglmsp.data.data[:, 0, :, :]
for ii in range(spectra.shape[0]):

    pxx = spectra[ii, :, :].mean(axis=0)
    pf = 0
    pp = 0
    sens = 0
    for jj in range(spectra.shape[1]):
        pf_tmp, pp_tmp = peak_find(gglmsp.f, spectra[ii, jj, :])
        if pp_tmp > pp:
            pf = pf_tmp
            pp = pp_tmp
            sens = jj

    nearest_f = np.argmin(np.abs(gglmsp.f - pf))
    topo = spectra[ii, :, nearest_f]

    pf2, pp2 = peak_find(gglmsp.f, spectra[ii, :, :].mean(axis=0))
    nearest_f2 = np.argmin(np.abs(gglmsp.f - pf2))

    info.append({'age': gglmsp.data.info['Age'][ii],
                 'peak_sens': sens,
                    'topo': topo,
                    'alpha_freq': pf,
                    'alpha_freq2': pf2,
                    'alpha_mag': pp,
                    'alpha_mag2': pp2})

df = pd.DataFrame(info)

topos = np.vstack(df.topo)
goods = np.logical_and(df.alpha_freq > 5, df.alpha_mag2 > 0)
topos = topos[goods, :]
freqs = df.alpha_freq[goods]
freqs2 = df.alpha_freq2[goods]
ages = df.age[goods]
mags = df.alpha_mag[goods]
mags2 = df.alpha_mag2[goods]


#%% ------------------------------------------
# Generate GLM interpolated alpha peak effect

I = np.argsort(ages.values)
f = gglmsp.f
colors = plt.cm.jet(np.linspace(0,1,len(I)))
    

nsteps_age = len(I)
age_spec, linelabels = gglmsp.model.project_range(1, nsteps=nsteps_age)

blah = np.zeros((3, len(I)))
blah2 = np.zeros((3, len(I)))
for ii in range(len(I)):
    pp = peak_find(f, spectra[goods,...][I[ii], :, :].mean(axis=(0)))
    blah[:, ii] = (ages.values[I[ii]], pp[0], pp[1])

    pp = peak_find(f, age_spec[ii, 0, :, :].mean(axis=0))
    blah2[:, ii] = (ages.values[I[ii]], pp[0], pp[1])

goods = blah[2, :] > 0
blah = blah[:, goods]


from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(blah[0, :], blah[1, :])
line_freq = slope * np.linspace(18, 88) + intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(blah[0, :], blah[2, :])
line_amp = slope * np.linspace(18, 88) + intercept

#%% ------------------------------------
# Figure 2

nsteps_age = 4
age_spec2, linelabels = gglmsp.model.project_range(1, nsteps=nsteps_age)

cmfun = plt.cm.cividis
colors = [(0.2, 0.2, 0.2), (0.9, 0, 0.9)] # first color is black, last is red
cmfun = LinearSegmentedColormap.from_list("Custom", colors, N=20)
cm = cmfun(np.linspace(0, 1, nsteps_age))
cm_dots = cmfun(np.linspace(0, 1, len(I)))

fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(3, 2, (1, 3))
for ii in range(nsteps_age):
    plt.plot(fx[0], age_spec2[ii, 0, :, :].mean(axis=0), lw=2, color=cm[ii, :])
osl.glm.decorate_spectrum(plt.gca(), ylabel='Relative Magnitude (a.u.)')
plt.xticks(np.sqrt(np.arange(4, 17)), np.arange(4, 17))
plt.xlabel('')
plt.xlim(2, 4)
plt.ylim(0.0003)
l = plt.legend([str(int(l)) + '' for l in linelabels], frameon=False, fontsize=11, ncol=4,
                bbox_to_anchor=(0.5, 0.95), loc="center",  bbox_transform=ax.transAxes)
l.set_title('Age (Years)')
#ax.set_facecolor((1.0, 0.47, 0.42))
subpanel_label(ax, 'A', title='GLM predicted individual alpha peak')
for ii in range(len(I)):
    plt.plot(np.sqrt(blah2[1, ii]), blah2[2, ii], 'o', color=cm_dots[ii])

ax = plt.subplot(3, 2, 5)
osl.glm.plot_sensor_spectrum(gglmsp.f, gglmsp.model.copes[1, 0, :, :].T, gglmsp.info, base=0.5, ax=ax, sensor_proj=True)
plt.xticks(np.sqrt(np.arange(4, 17)), np.arange(4, 17))
plt.xlim(2, 4)
plt.ylim(-0.0002, 0.0002)
plt.ylabel('Age Parameter Estimate')
subpanel_label(ax, 'B', title='Age Parameter Estimates')

ax = plt.subplot(222)
sc = scatter_density(blah[0, :], blah[1, :], 5, ax)
ax.plot(np.linspace(18, 88), line_freq, 'r', lw=2, label='Direct Peak Regression')
ax.plot(blah2[0, :], blah2[1, :], 'b', lw=2, label='GLM-Spectrum Predicted Peak')
plt.legend(frameon=False)
plt.ylim(6, 16)
for tag in ['top', 'right']:
    ax.spines[tag].set_visible(False)
ax.set_ylabel('Individual Alpha Frequency (Hz)')
ax.set_xlabel('Age (Years)')
subpanel_label(ax, 'C', title='Individual alpha peak frequency')

ax = plt.subplot(224)
sc = scatter_density(blah[0, :], blah[2, :], 5, ax)
ax.plot(np.linspace(18, 88), line_amp, 'r', lw=2, label='Direct Peak Regression')
ax.plot(blah2[0, :], blah2[2, :], 'b', lw=2, label='GLM-Spectrum Predicted Peak')
for tag in ['top', 'right']:
    ax.spines[tag].set_visible(False)
ax.set_ylabel('Individual Alpha Peak Magnitude')
ax.set_xlabel('Age (Years)')
subpanel_label(ax, 'D', title='Individual alpha peak magnitude')

plt.subplots_adjust(hspace=0.4, wspace=0.4)
figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure2.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))




#%% ------------------------------------------------
# Figure 3

log.info('Plotting figure 3 with stats table')

fig = plt.figure(figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95)
ax1 = plt.subplot(221)

freqs = (5, 8, 10.5, 15, 36, 60)
osl.glm.plot_joint_spectrum(gglmsp.f, cf2.T, gglmsp.info, 
                            ax=ax1, freqs=freqs, 
                            base=0.5, ylabel="Cohen's $F^2$")
subpanel_label(ax1, 'A', title="Cohen's $F^2$ effect size of Age", yf=0.8)

cf_line2 = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
ss_ticks = np.rint(glm_sample_size_calculation(cf_line2, 2, 0.05, 0.8)).astype(int)
secax = ax1.get_children()[11].secondary_yaxis('right', transform=ax1.transData)
secax.set_ylabel('Sample Size for 80% Power')
secax.set_ticks(cf_line2, ss_ticks)

peak_sens = []
info = []
for ii in range(len(freqs)):
    ax1.get_children()[12+ii].set_title('\n' + str(freqs[ii]) + 'Hz', fontsize=8, loc='center', )
    closest_f = np.abs(gglmsp.f - freqs[ii]).argmin()
    peak_sens_f = cf2[:, closest_f].argmax()
    peak_sens.append(gglmsp.info.ch_names[peak_sens_f])
    info.append([freqs[ii], 
                gglmsp.model.tstats[1, 0, peak_sens_f, closest_f],
                gglmsp.info.ch_names[peak_sens_f], 
                cf2[peak_sens_f, closest_f], 
                cf2_bs[2, peak_sens_f, closest_f],
                cf2_bs[-2, peak_sens_f, closest_f], 
                glm_sample_size_calculation(cf2_bs[2, peak_sens_f, closest_f], 2, 0.05, 0.8, 1e6),
                glm_sample_size_calculation(cf2_bs[-2, peak_sens_f, closest_f], 2, 0.05, 0.8, 1e6),
                ])

from tabulate import tabulate

headers = ['Peak Freq', 'Peak t-stat', 'Peak chan', 'Peak CF2', 'Lower CI', 'Upper CI', 'Lower sample', 'Upper sample']
print(tabulate(info, headers=headers))

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-table3.txt')
with open(figpath, 'w') as f:
    f.write(tabulate(info, headers=headers))
log.info('Saved table : {0}'.format(figpath))

axins = ax.inset_axes([0.75, 0.3, 0.35, 0.25])
osl.glm.plot_channel_layout(axins, gglmsp.info)

xy = None
for xx in ax.get_children():
    if xx.get_label() == 'inset_axes':
        if xx.get_xaxis().label.get_text() == 'Frequency (Hz)':
            xy = xx


def plot_channel_layout_tweak(ax, info, idx, size=30, marker='o'):
    ax.set_adjustable('box')
    ax.set_aspect('equal')
    colors, pos, outlines = get_mne_sensor_cols(info)
    pos_x, pos_y = pos.T
    mne.viz.evoked._prepare_topomap(pos, ax, check_nonzero=False)
    ax.scatter(pos_x, pos_y,
               color='k', s=10 * .8,
               marker='.', zorder=1)
    ax.scatter(pos_x[idx], pos_y[idx],
               color=colors[idx, :], s=size * 2,
               marker=marker, zorder=1)
    mne.viz.evoked._draw_outlines(ax, outlines)

sens = [10, 25, 50, 100]
sens = ['MEG1441', 'MEG2031', 'MEG2131', 'MEG1041']
sens = peak_sens[:4]
#freqs = [4, 8, 10, 16]
subpanels = ['i', 'ii', 'iii', 'iv']
colors, pos, outlines = get_mne_sensor_cols(gglmsp.info)

for ii in range(len(sens)):
    idx = mne.pick_channels(gglmsp.info.ch_names, [sens[ii]])[0]
    print(cf2[idx, :].max())

    ax = plt.subplot(2, 4, 5+ii)
    plt.plot(fx[0], cf2[idx, :], 'k')
    plt.fill_between(fx[0], cf2_bs[2, idx, :], cf2_bs[-2, idx, :], facecolor=colors[idx, :])
    plt.xticks(fx[2], fx[1])
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    plt.xlabel('Frequency (Hz)')
    if ii == 0:
        plt.ylabel("Cohen's $F^2$")
    plt.ylim(0, 0.4)
    subpanel_label(ax, 'C: ' + subpanels[ii], title='Peak sensor at {0}Hz'.format(freqs[ii]))

    axins = ax.inset_axes([0.6, 0.55, 0.35, 0.25])
    plot_channel_layout_tweak(axins, gglmsp.info, idx)

ax2 = plt.subplot(222, sharey=ax1.get_children()[11])
pos = ax2.get_position()
new_height = pos.height * (7/10)
ax2.set_position([pos.x0, pos.y0, pos.width, new_height])

labels = ['Power = {}'.format(nobs) for nobs in rel_power]
labels = ['{}'.format(nobs) for nobs in rel_power]
ll = plt.semilogx(sample_contours, cf_line, label=labels, color='k')
labelLines(ll, align=True, backgroundcolor="white", yoffsets=0.01, xvals=0.5*np.nanmean(sample_contours, axis=0))
ax2.xaxis.set_major_formatter(ScalarFormatter())
for tag in ['top', 'right']:
    ax2.spines[tag].set_visible(False)
ax2.set_xlabel('Sample Size')
ax2.set_ylabel("Cohen's $F^2$")

plt.xlim(5, 1000)
subpanel_label(ax2, 'B', yf = 1.2, xf=0,
               title='Sample Size-Effect Size contours of statistical power')

plt.grid(True, which='both')

figpath = os.path.join(bigmeg_meta['figure_dir'], 'bigmeg-camcan_draft-figure3.png')
fig.savefig(figpath, dpi=100, transparent=True)
log.info('Saved figure : {0}'.format(figpath))
