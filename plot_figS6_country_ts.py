"""
Plot time series of cancers in vaccination cohort by country
"""

import sciris as sc
import locations
import utils as ut
import numpy as np
from locations import locations, location_labels
 

def plot_single(ax, mres, color, label=None, smooth=False):

    years = np.arange(2024, 2126)  # 2024 to 2125
    res = mres['raw_cohort_cancers']
    best = np.cumsum(np.quantile(res, q=0.5, axis=-1))
    low = np.cumsum(np.quantile(res, q=0.1, axis=-1))
    high = np.cumsum(np.quantile(res, q=0.9, axis=-1))

    if smooth:
        best = np.convolve(list(best), np.ones(5), "valid")/5
        low = np.convolve(list(low), np.ones(5), "valid")/5
        high = np.convolve(list(high), np.ones(5), "valid")/5
        years = years[4:]

    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.1, color=color)
    return ax


def plot_cohort_ts():

    ut.set_font(16)
    n_plots = len(locations)+1  # +1 for the dummy plot
    fig, axes = sc.getrowscols(n_plots, make=True, remove_extra=True, figsize=(12, 10))
    axes = axes.flatten()

    colors = sc.gridcolors(3)

    for pn, location in enumerate(locations):
        ax = axes[pn]
        dflocation = location.replace(' ', '_')
        msim_dict = sc.loadobj(f'results/{dflocation}_vx_scens.obj')
        cn = 0
        for slabel, mres in msim_dict.items():
            ax = plot_single(ax, mres, color=colors[cn], label=slabel)
            cn += 1

        if pn in [0, 4, 8, 12, 16]: ax.set_ylabel('Cumulative cancers', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_title(location_labels[location])
        ax.legend().remove()  # Remove legend
        sc.SIticks(ax)

    # Make a dummy plot for the legend
    ax = axes[pn+1]
    for cn, slabel in enumerate(msim_dict.keys()):
        ax.plot(np.nan, np.nan, color=colors[cn], label=slabel)
    ax.axis('off')
    ax.legend(loc='center', fontsize=16, frameon=False)

    fig.tight_layout()
    fig_name = 'figures/figS6_country_ts.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    plot_cohort_ts()
