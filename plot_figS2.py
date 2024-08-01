"""
Plot ASR
"""
import hpvsim as hpv
import pylab as pl
import pandas as pd
import numpy as np
import sciris as sc
import utils as ut
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter



# Imports from this repository
import locations as loc
import utils as ut


#%% Plotting functions
def plot_figS2(locations):

    ut.set_font(14)
    colors = sc.gridcolors(2)

    n_plots = len(locations)
    fig, axes = sc.getrowscols(n_plots, make=True, remove_extra=True, figsize=(25,12))
    axes = axes.flatten()
    resname = 'asr_cancer_incidence'
    plot_count = 0
    date = 2020

    for pn, location in enumerate(locations):

        # Plot settings
        ax = axes[plot_count]
        dflocation = location.replace(' ', '_')
        res = sc.loadobj(f'raw_results/{dflocation}.mres')
        data = pd.read_csv(f'data/{dflocation}_asr_cancer_incidence.csv')

        start_year = 2000
        ind = sc.findinds(res['year'], start_year)[0]
        years = res['year'][ind:]

        ax.plot(years, res[resname].values[ind:], color=colors[0], label=f'HPVsim')
        ax.fill_between(years, res[resname].low[ind:],
                                res[resname].high[ind:], color=colors[0], alpha=0.3)

        ax.plot(2020, data['value'].values[0], marker='s', color=colors[1], label='Globocan')
        ax.set_ylabel('ASR incidence (per 100k)')
        ax.legend()
        ax.set_ylim(bottom=0)

        plot_count += 1

    fig.tight_layout()
    pl.savefig(f"figures/figS2.png", dpi=100)



# %% Run as a script
if __name__ == '__main__':

    plot_figS2(loc.locations)

    print('Done.') 
