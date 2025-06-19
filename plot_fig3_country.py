"""
Plot 1 for infant vaccination scenarios
"""


import pylab as pl
import pandas as pd
import sciris as sc
import locations as loc
import utils as ut
import numpy as np


def plot_fig2():

    ut.set_font(24)
    fig = pl.figure(figsize=(15, 10))
    colors = sc.gridcolors(3)

    df = pd.read_csv('results_direct.csv')

    # Plot cumulative cases by location
    df['Differences'] = df['Double dose'] - df['Single dose']
    dfsorted = df.sort_values(by='Differences', ascending=False)
    xlabels = [loc.location_labels[k] for k in dfsorted['location']]
    x = np.arange(len(xlabels))
    y = dfsorted['Differences']
    ax = fig.add_subplot(111)
    ax.bar(x-0.2, y, width=0.4, color=colors[2])
    # Rotate x axis labels by 90 degrees
    pl.setp(ax.get_xticklabels(), rotation=90, ha='center')
    ax.tick_params(axis='y', labelcolor=colors[2])
    ax.set_title('Additional girls vaccinated and cervical cancers averted\nin 2023/24 vaccination cohort by country')
    sc.SIticks(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    # Plot number of additional girls vaccinated
    ax2 = ax.twinx()
    y2 = [loc.vx_coverage_extra[k] for k in dfsorted['location']]
    # Add to plot
    ax2.bar(x+0.2, y2, width=0.4, color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    sc.SIticks(ax2)

    fig.tight_layout()
    fig_name = 'figures/fig3_country.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    plot_fig2()




