"""
Plot 1 for infant vaccination scenarios
"""


import pylab as pl
import pandas as pd
import sciris as sc
import locations as loc
import utils as ut
import numpy as np


def plot_fig3():

    ut.set_font(24)
    fig, axes = pl.subplots(3, 1, figsize=(15, 15))
    axes = axes.ravel()
    colors = sc.gridcolors(3)
    width = 0.6

    df = pd.read_csv('results_direct.csv')

    # Plot cumulative cases by location
    df['Differences'] = df['Double dose'] - df['Single dose']
    dfsorted = df.sort_values(by='Differences', ascending=False)
    xlabels = [loc.location_labels[k] for k in dfsorted['location']]
    x = np.arange(len(xlabels))
    y = dfsorted['Differences']

    # Plot cervical cancers averted
    ax = axes[0]
    ax.bar(x, y, width=width, color=colors[0])
    # Rotate x axis labels by 90 degrees
    ax.set_yscale('log')
    ax.set_title('Cervical cancers averted in 2023/24 cohort by country')
    top = 200e3
    ax.set_ylim(bottom=0, top=top)
    # Add text labels above bars
    for i, v in enumerate(y):
        ax.text(i, v*1.1, sc.sigfig(v, 3,sep=True), ha='center', va='bottom', fontsize=14)

    sc.SIticks(ax)
    ax.set_xticklabels([])  # Hide x labels for top plot

    # Plot number of additional girls vaccinated
    ax = axes[1]
    y2 = [loc.vx_coverage_extra[k] for k in dfsorted['location']]
    # Bar plot with log scale on y axis
    ax.bar(x, y2, width=width, color=colors[1])
    ax.set_yscale('log')
    ax.set_title('Additional girls vaccinated in 2023/24 cohort by country')
    sc.SIticks(ax)
    ax.set_xticklabels([])  # Hide x labels for top plot
    top = 12e6
    ax.set_ylim(bottom=0, top=top)
    # Add text labels above bars
    for i, v in enumerate(y2):
        ax.text(i, v*1.1, sc.sigfig(v, 3, sep=True), ha='center', va='bottom', fontsize=14)

    # Plot number needed to vaccinate (NNV), ratio of the two above
    ax = axes[2]
    nnv = np.array(y2) / y
    ax.bar(x, nnv, width=width, color=colors[2])
    # Set x label rotated 90 degrees
    ax.set_xticks(x, xlabels, rotation=90, ha='center')
    ax.set_title('Number needed to vaccinate (NNV) to avert one cervical cancer')
    # pl.setp(ax.get_xticklabels(), rotation=90, ha='center')
    top = 300
    ax.set_ylim(bottom=0, top=top)
    # Add a horizontal line at y=mean
    ax.axhline(np.sum(y2)/np.sum(y), color='k', ls='--', lw=0.5)
    # Add text labels above bars
    for i, v in enumerate(nnv):
        ax.text(i, v*1.05, sc.sigfig(v, 2, sep=True), ha='center', va='bottom', fontsize=14)

    fig.tight_layout()
    fig_name = 'figures/fig3_country.png'
    sc.savefig(fig_name, dpi=300)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    plot_fig3()




