"""
Plot 1 for infant vaccination scenarios
"""


import pylab as pl
import sciris as sc
import locations
import utils as ut
import numpy as np


def plot_single(ax, mres, to_plot, si, ei, color, label=None, smooth=True):
    years = mres.year[si:ei]
    ts = 0.5  # detection rate / sensitivity

    if to_plot == 'precin_incidence':
        best = mres.n_precin_by_age[3:10, si:ei].sum(axis=0) / mres.n_females_alive_by_age[3:10, si:ei].sum(axis=0) * ts
        low = mres.n_precin_by_age.low[3:10, si:ei].sum(axis=0) / mres.n_females_alive_by_age.low[3:10, si:ei].sum(axis=0) * ts
        high = mres.n_precin_by_age.high[3:10, si:ei].sum(axis=0) / mres.n_females_alive_by_age.high[3:10, si:ei].sum(axis=0) * ts
    else:
        best = mres[to_plot][si:ei]
        low = mres[to_plot].low[si:ei]
        high = mres[to_plot].high[si:ei]

    if smooth:
        best = np.convolve(list(best), np.ones(5), "valid")/5
        low = np.convolve(list(low), np.ones(5), "valid")/5
        high = np.convolve(list(high), np.ones(5), "valid")/5
        years = years[4:]

    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.1, color=color)
    return ax


def plot_fig1(locations):

    ut.set_font(16)
    n_plots = len(locations)
    fig, axes = sc.getrowscols(n_plots, make=True, remove_extra=True, figsize=(25, 12))
    axes = axes.flatten()
    resname = 'asr_cancer_incidence'

    colors = sc.gridcolors(3)

    # What to plot
    start_year = 2016
    end_year = 2100

    for pn, location in enumerate(locations):
        dflocation = location.replace(' ', '_')
        msim_dict = sc.loadobj(f'raw_results/{dflocation}_vx_scens.obj')
        mbase = msim_dict['Baseline']
        si = sc.findinds(mbase.year, start_year)[0]
        ei = sc.findinds(mbase.year, end_year)[0]
        ax = axes[pn]

        cn = 0
        for slabel, mres in msim_dict.items():
            ax = plot_single(ax, mres, resname, si, ei, color=colors[cn], label=slabel)
            cn += 1

        ax.set_ylim(bottom=0)  #, top=23)
        ax.set_ylabel(resname)
        ax.set_title(location.title())
        if pn == 0: ax.legend(frameon=False)
        if resname == 'asr_cancer_incidence': ax.axhline(y=4, color='k', ls='--')

    fig.tight_layout()
    fig_name = 'figures/fig1_vx_scens.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    locations = locations.locations
    plot_fig1(locations)




