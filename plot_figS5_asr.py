"""
Fig S5: per-country ASR cancer-incidence time series vs Globocan 2020.

Reads `figS5_asr.csv` (per-country TS) + `data/<loc>_asr_cancer_incidence.csv` (Globocan).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc

import locations as loc
import utils as ut


def plot_figS5(locations, resfolder='results/v2.2.6_baseline',
               datafolder='data', outpath='figures/figS5_asr.png'):
    asr_df = pd.read_csv(f'{resfolder}/figS5_asr.csv')

    ut.set_font(16)
    colors = sc.gridcolors(2)
    n_plots = len(locations)
    fig, axes = sc.getrowscols(n_plots, make=True, remove_extra=True, figsize=(12, 10))
    axes = axes.flatten()

    for pn, location in enumerate(locations):
        ax = axes[pn]
        sub = asr_df[asr_df.location == location].sort_values('year')
        if sub.empty:
            continue
        years = sub.year.values
        ax.plot(years, sub.value.values, color=colors[0], label='HPVsim')
        ax.fill_between(years, sub.low.values, sub.high.values,
                        color=colors[0], alpha=0.3)

        dfl = location.replace(' ', '_')
        try:
            data = pd.read_csv(f'{datafolder}/{dfl}_asr_cancer_incidence.csv')
            ax.plot(2020, data['value'].values[0], marker='s',
                    color=colors[1], label='Globocan')
        except FileNotFoundError:
            pass
        ax.set_ylabel('ASR incidence (per 100k)')
        ax.set_title(location.capitalize())
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    pl.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results/v2.2.6_baseline')
    parser.add_argument('--datafolder', default='data')
    parser.add_argument('--outpath', default='figures/figS5_asr.png')
    args = parser.parse_args()
    plot_figS5(loc.locations, resfolder=args.resfolder,
               datafolder=args.datafolder, outpath=args.outpath)
    print('Done.')
