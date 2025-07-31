import os
import glmtools
import pandas as pd
import numpy as np
import tomllib
import glmtools as glm
import osl
import mne
from scipy import signal, stats, spatial, optimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle 
from copy import deepcopy

with open("pyproject_paths.toml", "rb") as f:
    bigmeg_meta = tomllib.load(f)

log = osl.logging.getLogger()
log.setLevel('INFO')

#%% 
# General stats

def rank_biserial_correlation(U, n1, n2):
    return 1 - (2*U) / (n1 * n2)


def pstring(pvalue, thresh=0.001):
    if pvalue < thresh:
        return 'p<{}'.format(thresh)
    else:
        return 'p={0:.3f}'.format(pvalue)


def glm_power_calculation(f2, num_regressors, num_observations, siglevel):
    """
    Calculate the power of an effect from a general linear model (GLM) regression.
    
    """
    ncp = f2*(num_regressors+num_observations+1)
    tmp = stats.f.ppf(1 - siglevel, num_regressors, num_observations)
    return 1 - stats.ncf.cdf(tmp, num_regressors, num_observations, ncp)


def glm_sample_size_calculation(f2, num_regressors, siglevel, power, max_N=1000):
    """
    Calculate sample size required to achieve a given power 
    for general linear model (GLM) regression.
    
    """
    
    def func(num_observations, num_regressors, f2, siglevel, power):
        tmp = stats.f.ppf(1 - siglevel, num_regressors, num_observations)
        ncp = f2*(num_regressors+num_observations+1)
        return 1 - stats.ncf.cdf(tmp, num_regressors, num_observations, ncp) - power
    
    def run_brentq(func, min_N, max_N, args=None):
        try:
            return optimize.brentq(func, min_N, max_N, args=args)
        except ValueError:
            return np.nan

    if np.array(f2).size == 1:
        args = (num_regressors, f2, siglevel, power)
        return run_brentq(func, 1, max_N, args=args)
    else:
        # loop over items in input array
        f2_flat = np.array(f2).reshape(-1)
        sample_size = np.zeros_like(f2_flat)
        for ii in range(f2_flat.shape[0]):
            args = (num_regressors, f2_flat[ii], siglevel, power)
            sample_size[ii] = run_brentq(func, 1, max_N, args=args)
        
        return sample_size.reshape(f2.shape)


def glm_effect_size_calculation(num_observations, num_regressors, siglevel, power):
    """
    Calculate sample size required to achieve a given power 
    for general linear model (GLM) regression.
    
    """
    
    def func(f2, num_observations, num_regressors, siglevel, power):
        return glm_power_calculation(f2, num_regressors, num_observations, siglevel) - power

    def run_brentq(func, min_f2, max_f2, args=None):
        try:
            return optimize.brentq(func, min_f2, max_f2, args=args)
        except ValueError:
            return np.nan

    if np.array(num_observations).size == 1:
        args = (num_observations, num_regressors, siglevel, power)
        print(args)
        return run_brentq(func, 0, 1, args=args)
    else:
        # loop over items in input array
        num_observations_flat = np.array(num_observations).reshape(-1)
        effect_size = np.zeros_like(num_observations_flat)
        for ii in range(num_observations_flat.shape[0]):
            args = (num_regressors, num_observations_flat[ii], siglevel, power)
            effect_size[ii] = run_brentq(func, 0, 1, args=args)

        return effect_size.reshape(num_observations.shape)

#%% 
# Data Proc

def load_headvol(df):
    base_outdir = os.path.join(bigmeg_meta['mri_output_dir'], "sub-{subj_id}/sub-{subj_id}_headvol_stats")
    st = osl.utils.Study(base_outdir)
    head = []
    skull = []
    brain = []

    for idx, subj_id in enumerate(df['ID'].values):
        try:
            x = np.loadtxt(st.get(subj_id=subj_id)[0])
            head.append(x[0, 1])
            skull.append(x[1, 1])
            brain.append(x[2, 1])
        except IndexError:
            head.append(np.nan)
            skull.append(np.nan)
            brain.append(np.nan) 
    
    df['NewHeadVol'] = head
    df['NewSkullVol'] = skull
    df['NewBrainVol'] = brain

    def nanz(xxx):
        return (xxx - np.nanmean(xxx)) / np.nanstd(xxx)

    df['HeadXSkull'] = nanz(head) * nanz(skull)
    df['HeadXBrain'] = nanz(head) * nanz(brain)

    return df


def load_cardio(df):

    cardio_file = os.path.join(bigmeg_meta['code_dir'], 'CardioMeasures_summary.xlsx')
    cardio = pd.read_excel(cardio_file)
    bpm = []
    systol = []
    diastol = []
    height = []
    weight = []

    for subj in df.ID:
        match_ind = np.where(cardio.CCID == subj)[0]
        if len(match_ind) == 1:
            bpm.append(cardio.iloc[match_ind[0]].pulse_mean)
            systol.append(cardio.iloc[match_ind[0]].bp_sys_mean)
            diastol.append(cardio.iloc[match_ind[0]].bp_dia_mean)
            height.append(cardio.iloc[match_ind[0]].height)
            weight.append(cardio.iloc[match_ind[0]].weight)
        else:
            bpm.append(np.nan)
            systol.append(np.nan)
            diastol.append(np.nan)
            height.append(np.nan)
            weight.append(np.nan)

    df['BPM'] = bpm
    df['BPSys'] = systol
    df['BPDia'] = diastol
    df['Height'] = height
    df['Weight'] = weight

    return df


def load_headpos(infiles, df):

    x = []
    y = []
    z = []
    r = []

    for ff in infiles:
        glmsp = osl.glm.read_glm_spectrum(ff)

        if 'head_pos' in glmsp.model.extras:
            hp = glmsp.model.extras['head_pos']
            r.append(glmsp.model.extras['head_radius'])
            x.append(hp[0])
            y.append(hp[1])
            z.append(hp[2])
        else:
            r.append(np.nan)
            x.append(np.nan)
            y.append(np.nan)
            z.append(np.nan)

    df['x'] = x
    df['y'] = y
    df['z'] = z
    df['head_radius'] = r

    return df


#%% 
# GLMS

def get_matched_vect(df, subj_ids, key):
    out = []
    for ii in range(len(subj_ids)):
        sid = subj_ids[ii]
        sid = sid[4:] if sid.startswith('sub-') else sid
        inds = np.where(np.array([row_id.find(sid) for row_id in df['ID'].values]) > -1)[0]
        if len(inds) > 0:
            out.append(df[key].iloc[inds[0]])
        else:
            out.append(np.nan)
    return out


def load_matched_from_csv(subj_id, df):

    if isinstance(df, str):
        df = pd.read_csv(df)

    info = {}

    info['age'] = get_matched_vect(df, subj_id, 'Fixed_Age')
    info['age2'] = list(np.array(info['age'])**2)
    info['sex'] = get_matched_vect(df, subj_id, 'Sex (1=female, 2=male)')
    info['Brain_Vol'] = get_matched_vect(df, subj_id, 'Brain_Vol')
    info['GM_Vol_Norm'] = get_matched_vect(df, subj_id, 'GM_Vol_Norm')
    info['WM_Vol_Norm'] = get_matched_vect(df, subj_id, 'WM_Vol_Norm')
    info['Hippo_Vol_Norm'] = get_matched_vect(df, subj_id, 'Hippo_Vol_Norm')

    info['NewHeadVol'] = get_matched_vect(df, subj_id, 'NewHeadVol')
    info['NewSkullVol'] = get_matched_vect(df, subj_id, 'NewSkullVol')
    info['NewBrainVol'] = get_matched_vect(df, subj_id, 'NewBrainVol')

    info['HeadXSkull'] = get_matched_vect(df, subj_id, 'HeadXSkull')
    info['HeadXBrain'] = get_matched_vect(df, subj_id, 'HeadXBrain')

    info['BPM'] = get_matched_vect(df, subj_id, 'BPM')
    info['BPSys'] = get_matched_vect(df, subj_id, 'BPSys')
    info['BPDia'] = get_matched_vect(df, subj_id, 'BPDia')
    info['Height'] = get_matched_vect(df, subj_id, 'Height')
    info['Weight'] = get_matched_vect(df, subj_id, 'Weight')
    #info['BMI'] = np.array(info['Weight']) / ((np.array(info['Height']) / 100)**2)

    return info


def exclude_for_missing(infiles, info):
    keeps = np.ones_like(infiles, dtype=bool)
    for key, item in info.items():
        keeps[np.isnan(item)] = False
    
    log.info('Missing: {} in - {} kept'.format(len(infiles), np.sum(keeps)))

    for key, item in info.items():
        info[key] = np.array(item)[keeps]
    
    outfiles = np.array(infiles)[keeps]

    return outfiles, info


def exclude_for_leverage(infiles, info, DC):
    covs = deepcopy(info)
    covs['num_observations'] = len(covs['age'])
    design = DC.design_from_datainfo(covs)
    keeps = design.leverage < 5*np.median(design.leverage)

    log.info('Leverage: {} in - {} kept'.format(len(infiles), np.sum(keeps)))

    for key, item in info.items():
        info[key] = np.array(item)[keeps]
    
    outfiles = np.array(infiles)[keeps]

    return outfiles, info


def exclude_for_value(infiles, info, key, value):
    keeps = info[key] > value

    log.info('{}: {} in - {} kept'.format(key, len(infiles), np.sum(keeps)))

    for key, item in info.items():
        info[key] = np.array(item)[keeps]
    
    outfiles = np.array(infiles)[keeps]

    return outfiles, info


def exclude_for_hippovol(infiles, info):
    keeps = info['Hippo_Vol_Norm'] > 0.355

    log.debug('Hippocampus: {} in - {} kept'.format(len(infiles), np.sum(keeps)))

    for key, item in info.items():
        info[key] = np.array(item)[keeps]
    
    outfiles = np.array(infiles)[keeps]

    return outfiles, info


def cohens_f2(glmsp, reg_idx=0, mode='drop'):
    from glmtools.fit import OLSModel
    cf2 = np.zeros_like(glmsp.model.betas)
    denom = 1 - glmsp.model.r_square

    small_design = deepcopy(glmsp.design)

    small_design.design_matrix = np.delete(small_design.design_matrix, reg_idx, axis=1)
    small_design.contrasts = np.delete(small_design.contrasts, reg_idx, axis=1)
    rname = small_design.regressor_names.pop(reg_idx)
    reg = small_design.regressor_list.pop(reg_idx)
    log.debug("Dropping '{0}'".format(rname))

    small_model = OLSModel(design=small_design, data_obj=glmsp.data)
    log.debug("Full R-square: {0}, Small R-square: {1}".format(glmsp.model.r_square[0, 0, :, :].mean(),
                                                               small_model.r_square[0, 0, :, :].mean()))

    cf2 = (glmsp.model.r_square - small_model.r_square) / denom
    log.debug("Cohen's F2 : {0}", cf2[0, 0, :, :].mean())

    return cf2


#%% 
# Plotting


def subpanel_label(ax, label, title='', xf=-0.1, yf=1.1, ha='center'):
    ypos = ax.get_ylim()[0]
    yyrange = np.diff(ax.get_ylim())[0]
    ypos = (yyrange * yf) + ypos
    # Compute letter position as proportion of full xrange.
    xpos = ax.get_xlim()[0]
    xxrange = np.diff(ax.get_xlim())[0]
    xpos = (xxrange * xf) + xpos
    ax.text(xpos, ypos, label, horizontalalignment=ha,
            verticalalignment='center', fontsize=20, fontweight='bold')
    ax.set_title(title, loc='right', fontsize=14)


def scatter_density(x, y, q, ax, title=None):
    # Calculate the point density
    xy = np.vstack([x,y])
    z = stats.gaussian_kde(xy)(xy)

    if isinstance(q, int) is False:
        q = (q - np.min(q)) / (np.max(q) - np.min(q)) * 100

    sc = ax.scatter(x, y, c=z, s=q)

    if title is not None:
        ax.set_title(title)

    return sc


def plot_sig_clusters_with_map(gglmsp, P, thresh, base=1,
                               outbase=None, nclusters=5, yl=None,
                               stitle='', I=None):
    from osl.glm.glm_spectrum import plot_joint_spectrum_clusters, decorate_spectrum, plot_sensor_spectrum
    from matplotlib.patches import ConnectionPatch
    
    #%% Setup
    from mne.stats.cluster_level import _find_clusters as mne_find_clusters
    from mne.stats.cluster_level import _reshape_clusters as mne_reshape_clusters

    gtitle = 'group con : {}'.format(gglmsp.contrast_names[P.gl_con])
    ftitle = 'first-level con : {}'.format(gglmsp.fl_contrast_names[P.fl_con])
    stitle = gtitle + '\n' + ftitle

    obs = gglmsp.model.get_tstats()[P.gl_con, P.fl_con, :, :]  # [Chans x Freqs]
    thresh = P.perms.get_thresh(99)

    obs_up = obs > thresh
    obs_down = obs < -thresh
    thresh_obs = obs_up.astype(int) - obs_down.astype(int)

    clus_up, cstat_up = mne_find_clusters(obs.T.flatten(), thresh, adjacency=P.adjacency)
    clus_up = mne_reshape_clusters(clus_up, obs.T.shape)

    nclusters = len(clus_up) if len(clus_up) < nclusters else nclusters
    if nclusters == 0:
        clusters = []
    else:
        cthresh = np.sort(np.abs(cstat_up))[-nclusters]
        clusters = [(x, 0, (c[0], c[1])) for c, x in zip(clus_up, cstat_up) if np.abs(x) >= cthresh]

    lay = mne.channels.read_layout('Vectorview-mag')
    I = np.argsort(lay.pos[:, 1])

    log.debug('thresh_obs : {}'.format(thresh_obs.shape))
    log.debug('obs : {}'.format(obs.shape))
    log.debug('glmsp.f : {}'.format(gglmsp.f.shape))
    log.debug('I : {}'.format(I.shape))

    fx = osl.glm.prep_scaled_freq(base, gglmsp.f)

    #%% Start plotting

    thresh_sig = np.ma.masked_array(obs, np.abs(obs) < thresh)
    thresh_insig = np.ma.masked_array(obs, np.abs(obs) > thresh)

    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(stitle)
    plt.subplots_adjust(hspace=0.4)
    ax2 = plt.subplot(9, 1, (8, 9))
    X = np.r_[fx[0] - np.diff(fx[0])[0]/2, fx[0][-1] + np.diff(fx[0])[-1]]
    ax2.pcolormesh(X, np.arange(103), thresh_obs[I, :])
    ax2.set_xticks(fx[2], fx[1])
    ax2.set_yticks([0, 102], ['Posterior', 'Anterior'])
    ax2.set_xlabel('Frequency (Hz)')

    ax1 = plt.subplot(9, 1, (3, 6))
    plot_sensor_spectrum(gglmsp.f, obs.T, gglmsp.info, ax=ax1, base=base, lw=0.25, ylabel='t-statistics')

    if yl is not None:
        ax1.set_ylim(yl)

    ymax_span = ax1.get_ylim()[1]
    ax1.set_xlim(X[0], X[-1])

    log.debug('Obs shape - {}'.format(obs.shape))

    # Reorder clusters in ascending frequency
    clu_order = []
    for clu in clusters:
        clu_order.append(clu[2][0].min())
    clusters = [clusters[ii] for ii in np.argsort(clu_order)]

    yl = ax1.get_ylim()

    topo_centres = np.linspace(0, 1, len(clusters)+2)[1:-1]
    cinfo = []

    for idx, clu in enumerate(clusters):
        cfreqs, cchans  = clu[2]

        assert(cchans.max() < 102)

        # Extract cluster location in space and frequency
        channels = np.zeros((obs.shape[0], ))
        channels[cchans] = 1
        if len(channels) == 204:
            channels = np.logical_or(channels[::2], channels[1::2])
        log.debug('channels - {}'.format(channels))
        log.debug('channels shape - {}'.format(channels.shape))

        finds = np.unique(cfreqs)
        log.debug('finds - {}'.format(finds))
        log.debug('finds - {}'.format(fx[0][finds]**2))

        freq_range = (fx[0][finds[0]]**2, fx[0][finds[-1]]**2)
        avgt = clu[0]/len(cfreqs)
        avgp = 1 - (stats.percentileofscore(P.perms.nulls, np.abs(avgt)) / 100)
        cinfo.append({'Cluster': idx + 1,
                      'Statistic': clu[0],
                      'Average T': avgt,
                      'DoF': gglmsp.model.dof_model,
                      'Nperms': P.perms.nperms,
                      'Average p': avgp,
                      'Min Freq': freq_range[0],
                      'Max Freq': freq_range[-1],
                      'Num Channels': int(channels.sum())})

        # Plot cluster span overlay on spectrum
        ax1.axvspan(fx[0][finds[0]], fx[0][finds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5, ymax=ymax_span)
        
        topo_x = topo_centres[idx] #  fx[0][finds].mean() / fx[0].max()
        log.debug('xcoord - {}'.format(topo_x))
        topo_pos = [topo_x - 0.1, 1.05, 0.2, 0.35]
        topo_ax = ax1.inset_axes(topo_pos)

        # Plot topo
        topomap_args = {}
        dat = obs[:, finds].mean(axis=1)
        im, cn = mne.viz.plot_topomap(dat, gglmsp.info, 
                                      axes=topo_ax, show=False, 
                                      mask=channels, ch_type='mag', 
                                      contours=0,
                                      **topomap_args)

        # Plot connecting line to topo
        xy_main = (fx[0][finds].mean(), yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=ax1.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        ax1.figure.add_artist(con)

    outpng = outbase + '.png' if outbase is not None else None

    if outpng is not None:
        plt.savefig(outpng, dpi=100)

    if outbase is not None:
        outcsv = outbase + '.csv'
        pd.DataFrame(cinfo).to_csv(outcsv)
        
    return fig, pd.DataFrame(cinfo)


def plot_sig_clusters_with_map2(gglmsp, P, thresh, base=1, yl=None,
                                outbase=None, nclusters=5, jitter_levels=2,
                                stitle='', I=None, specgrid=True,
                                ax1=None, ax2=None):
    from osl.glm.glm_spectrum import plot_joint_spectrum_clusters, decorate_spectrum, plot_sensor_spectrum
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    
    #%% Setup
    from mne.stats.cluster_level import _find_clusters as mne_find_clusters
    from mne.stats.cluster_level import _reshape_clusters as mne_reshape_clusters

    gtitle = 'group con : {}'.format(gglmsp.contrast_names[P.gl_con])
    ftitle = 'first-level con : {}'.format(gglmsp.fl_contrast_names[P.fl_con])
    stitle = gtitle + '\n' + ftitle

    obs = gglmsp.model.get_tstats()[P.gl_con, P.fl_con, :, :]  # [Chans x Freqs]
    thresh = P.perms.get_thresh(99)

    obs_up = obs > thresh
    obs_down = obs < -thresh
    thresh_obs = obs_up.astype(int) - obs_down.astype(int)

    clus_up, cstat_up = mne_find_clusters(obs.T.flatten(), thresh, adjacency=P.adjacency)
    clus_up = mne_reshape_clusters(clus_up, obs.T.shape)

    nclusters = len(clus_up) if len(clus_up) < nclusters else nclusters
    if nclusters == 0:
        clusters = []
    else:
        cthresh = np.sort(np.abs(cstat_up))[-nclusters]
        clusters = [(x, 0, (c[0], c[1])) for c, x in zip(clus_up, cstat_up) if np.abs(x) >= cthresh]

    lay = mne.channels.read_layout('Vectorview-mag')
    I = np.argsort(lay.pos[:, 1])

    log.debug('thresh_obs : {}'.format(thresh_obs.shape))
    log.debug('obs : {}'.format(obs.shape))
    log.debug('glmsp.f : {}'.format(gglmsp.f.shape))
    log.debug('I : {}'.format(I.shape))

    fx = osl.glm.prep_scaled_freq(base, gglmsp.f)

    thresh_sig = np.ma.masked_array(obs, np.abs(obs) < thresh)
    thresh_insig = np.ma.masked_array(obs, np.abs(obs) > thresh)
    vm = np.abs(obs).max()

    #%% Start plotting

    if ax1 is None:
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(stitle)
        plt.subplots_adjust(hspace=0.4)
    else:
        fig = ax1.get_figure()

    if ax2 is None:
        ax2 = plt.subplot(9, 1, (8, 9))

    X = np.r_[fx[0] - np.diff(fx[0])[0]/2, fx[0][-1] + np.diff(fx[0])[-1]]
    #ax2.pcolormesh(X, np.arange(103), thresh_obs[I, :])
    im1 = ax2.pcolormesh(X, np.arange(103), thresh_sig[I, :], vmin=-vm, vmax=vm, cmap='RdBu_r')
    ax2.pcolormesh(X, np.arange(103), thresh_insig[I, :], vmin=-vm, vmax=vm, cmap='RdBu_r', alpha=0.4)

    ax2.set_xticks(fx[2], fx[1])
    ax2.set_yticks([0, 102], ['Posterior', 'Anterior'])
    ax2.set_xlabel('Frequency (Hz)')

    pos = ax2.get_position().bounds
    cax2 = plt.axes([pos[0] + pos[2] + 0.01, pos[1], 0.02, pos[3]])
    cb2 = fig.colorbar(im1, cax=cax2)
    cb2.set_label('t-statistics') #, fontsize=8)

    #ax2_divider = make_axes_locatable(ax2)
    #cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")

    if ax1 is None:
        ax1 = plt.subplot(9, 1, (3, 6))
        ax1.grid(True)

    plot_sensor_spectrum(gglmsp.f, obs.T, gglmsp.info, ax=ax1, base=base, lw=0.25, ylabel='t-statistics')

    if yl is not None:
        ax1.set_ylim(yl)

    ymax_span = ax1.get_ylim()[1]
    ax1.set_xlim(X[0], X[-1])

    log.debug('Obs shape - {}'.format(obs.shape))

    # Reorder clusters in ascending frequency
    clu_order = []
    for clu in clusters:
        clu_order.append(clu[2][0].min())
    clusters = [clusters[ii] for ii in np.argsort(clu_order)]

    yl = ax1.get_ylim()

    topo_centres = np.linspace(0, 1, len(clusters)+2)[1:-1]
    cinfo = []

    jitters = 1 + (-8/100) * (np.arange(len(clusters)) % jitter_levels)

    for idx, clu in enumerate(clusters):
        cfreqs, cchans  = clu[2]

        #assert(cchans.max() < 102)

        # Extract cluster location in space and frequency
        channels = np.zeros((obs.shape[0], ))
        channels[cchans] = 1
        if len(channels) == 204:
            channels = np.logical_or(channels[::2], channels[1::2])
        log.debug('channels - {}'.format(channels))
        log.debug('channels shape - {}'.format(channels.shape))

        finds = np.unique(cfreqs)
        log.debug('finds - {}'.format(finds))
        log.debug('finds - {}'.format(fx[0][finds]**2))

        freq_range = (fx[0][finds[0]]**2, fx[0][finds[-1]]**2)
        avgt = clu[0]/len(cfreqs)
        avgp = 1 - (stats.percentileofscore(P.perms.nulls, np.abs(avgt)) / 100)
        cinfo.append({'Cluster': idx + 1,
                      'Statistic': clu[0],
                      'Average T': avgt,
                      'DoF': gglmsp.model.dof_model,
                      'Nperms': P.perms.nperms,
                      'Average p': avgp,
                      'Min Freq': freq_range[0],
                      'Max Freq': freq_range[-1],
                      'Num Channels': int(channels.sum())})

        # Plot cluster span overlay on spectrum
        #ax1.axvspan(fx[0][finds[0]], fx[0][finds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5, ymax=ymax_span)
        print(ymax_span)
        ym = (0.95 * ymax_span) * jitters[idx]
        print(ym)
        ax1.plot((fx[0][finds[0]], fx[0][finds[-1]]), (ym, ym), 'k')

        yspanh = np.ptp((yl[0], ym)) / np.ptp(yl)  # Needs to be in axis units

        ax1.axvspan(fx[0][finds[0]], fx[0][finds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.2, ymax=yspanh)

        topo_x = topo_centres[idx] #  fx[0][finds].mean() / fx[0].max()
        log.debug('xcoord - {}'.format(topo_x))
        #topo_pos = [topo_x - 0.1, 1.05, 0.2, 0.35]
        topo_pos = [topo_x - 0.1, 1.1, 0.2, 0.35]
        topo_ax = ax1.inset_axes(topo_pos)

        # Plot topo
        topomap_args = {}
        dat = obs[:, finds].mean(axis=1)
        im, cn = mne.viz.plot_topomap(dat, gglmsp.info, 
                                      axes=topo_ax, show=False, 
                                      mask=channels, ch_type='mag', 
                                      contours=0,
                                      **topomap_args)

        # Plot connecting line to topo
        xy_main = (fx[0][finds].mean(), yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=ax1.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        ax1.figure.add_artist(con)

        ax1.plot((fx[0][finds].mean(), fx[0][finds].mean()), (ym, ymax_span), color=[0.7, 0.7, 0.7])

    outpng = outbase + '.png' if outbase is not None else None

    if outpng is not None:
        plt.savefig(outpng, dpi=100)

    if outbase is not None:
        outcsv = outbase + '.csv'
        pd.DataFrame(cinfo).to_csv(outcsv)
        
    return fig, pd.DataFrame(cinfo)

#%% 
## Nonsense

class PatchedGroupSensorGLMSpectrum(osl.glm.GroupSensorGLMSpectrum):
    """
    For stupid monkey patch patching of mne.io.meas_info > mne._fiff.meas_info issue
    """
    def get_channel_adjacency(self, dist=40):

        if np.any(['parcel' in ch for ch in self.info['ch_names']]):
            # We have parcellated data
            parcellation_file = parcellation.guess_parcellation(int(np.sum(['parcel' in ch for ch in self.info['ch_names']])))
            adjacency = csr_array(parcellation.spatial_dist_adjacency(parcellation_file, dist=dist))
        elif np.any(['state' in ch for ch in self.info['ch_names']]) or np.any(['mode' in ch for ch in self.info['ch_names']]):
            adjacency = csr_array(np.eye(len(self.info['ch_names'])))
        else:
            #ch_type =  mne._fiff.meas_info.get_channel_types(self.info)[0]  # Assuming these are all the same!
            ch_type = self.info.get_channel_types()[0]
            adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(self.info, ch_type)
        ntests = np.prod(self.model.copes.shape[2:])
        ntimes = self.model.copes.shape[3]
        print('{} : {}'.format(ntimes, ntests))
        return mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)
