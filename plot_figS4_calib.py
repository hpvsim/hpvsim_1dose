"""
Fig S4: calibration boxplots (cancers by age) for each country.

Reads `figS4_calib.csv` (model trials) + `figS4_targets.csv` (Globocan) produced
by `utils.extract_figS4_calib_csvs`.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import seaborn as sns

import locations as loc
import utils as ut


def plot_figS4(locations, resfolder='results/v2.2.6_baseline',
               outpath='figures/figS4_calib.png'):
    model = pd.read_csv(f'{resfolder}/figS4_calib.csv')
    targets = pd.read_csv(f'{resfolder}/figS4_targets.csv')

    ut.set_font(16)
    n_plots = len(locations)
    fig, axes = sc.getrowscols(n_plots, make=True, remove_extra=True, figsize=(12, 10))
    axes = axes.flatten()

    for pn, location in enumerate(locations):
        ax = axes[pn]

        t_sub = targets[targets.location == location].sort_values('bin_idx')
        m_sub = model[model.location == location]

        if t_sub.empty or m_sub.empty:
            continue

        x = np.arange(len(t_sub))
        ax.scatter(x, t_sub.value.values, color='k', marker='s', label='Data')

        sns.boxplot(ax=ax, x='bin_idx', y='value', data=m_sub,
                    color='b', boxprops=dict(alpha=.4))

        title = location.title()
        if title == 'Cote Divoire':
            title = "Cote d'Ivoire"
        ax.set_title(title)
        ax.set_ylabel('')
        ax.set_xlabel('')
        if pn in [0, 4, 8, 12, 16]:
            ax.set_ylabel('# cancers')

        # Show every other age bin as x-tick
        bin_labels = t_sub.bin_label.values
        stride = np.arange(0, len(bin_labels), 2)
        ax.set_xticks(x[stride])
        ax.set_xticklabels([bin_labels[i].split('-')[0] if '-' in bin_labels[i]
                            else bin_labels[i].replace('+', '')
                            for i in stride])

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    pl.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results/v2.2.6_baseline')
    parser.add_argument('--outpath', default='figures/figS4_calib.png')
    args = parser.parse_args()
    plot_figS4(loc.locations, resfolder=args.resfolder, outpath=args.outpath)
    print('Done.')
