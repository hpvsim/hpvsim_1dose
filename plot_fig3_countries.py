"""
Fig 3: per-country cancers averted + girls vaccinated + NNV bar chart.

Reads `results_direct.csv` (from the committed baseline).
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


def plot_fig3(resfolder='results/v2.2.6_baseline',
              outpath='figures/fig3_country.png'):
    ut.set_font(24)
    fig, axes = pl.subplots(3, 1, figsize=(15, 15))
    axes = axes.ravel()
    colors = sc.gridcolors(3)
    width = 0.6

    df = pd.read_csv(f'{resfolder}/results_direct.csv')

    df['Differences'] = df['Double dose'] - df['Single dose actual']
    dfsorted = df.sort_values(by='Differences', ascending=False)
    xlabels = [loc.location_labels[k] for k in dfsorted['location']]
    x = np.arange(len(xlabels))
    y = dfsorted['Differences']

    # Panel A: cancers averted
    ax = axes[0]
    ax.bar(x, y, width=width, color=colors[0])
    ax.set_yscale('log')
    ax.set_title('Cervical cancers averted in 2023/24 cohort by country')
    ax.set_ylim(bottom=0, top=200e3)
    for i, v in enumerate(y):
        ax.text(i, v * 1.1, sc.sigfig(v, 3, sep=True), ha='center', va='bottom', fontsize=14)
    sc.SIticks(ax)
    ax.set_xticklabels([])

    # Panel B: additional girls vaccinated
    ax = axes[1]
    y2 = [loc.vx_coverage_extra[k] for k in dfsorted['location']]
    ax.bar(x, y2, width=width, color=colors[1])
    ax.set_yscale('log')
    ax.set_title('Additional girls vaccinated in 2023/24 cohort by country')
    sc.SIticks(ax)
    ax.set_xticklabels([])
    ax.set_ylim(bottom=0, top=12e6)
    for i, v in enumerate(y2):
        ax.text(i, v * 1.1, sc.sigfig(v, 3, sep=True), ha='center', va='bottom', fontsize=14)

    # Panel C: NNV
    ax = axes[2]
    nnv = np.array(y2) / y
    ax.bar(x, nnv, width=width, color=colors[2])
    ax.set_xticks(x, xlabels, rotation=90, ha='center')
    ax.set_title('Number needed to vaccinate (NNV) to avert one cervical cancer')
    ax.set_ylim(bottom=0, top=300)
    ax.axhline(np.sum(y2) / np.sum(y), color='k', ls='--', lw=0.5)
    for i, v in enumerate(nnv):
        ax.text(i, v * 1.05, sc.sigfig(v, 2, sep=True), ha='center', va='bottom', fontsize=14)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    sc.savefig(outpath, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results/v2.2.6_baseline')
    parser.add_argument('--outpath', default='figures/fig3_country.png')
    args = parser.parse_args()
    plot_fig3(resfolder=args.resfolder, outpath=args.outpath)
    print('Done.')
