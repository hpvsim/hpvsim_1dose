"""
Cross-version comparison for hpvsim_1dose.

Overlays the cumulative-cancers time series from fig2_res_stats.csv across
baseline versions, and tabulates the cumulative 2025-2125 values per scenario.

Usage:
  python compare_baselines.py --baselines v2.2.6_baseline v2.3.0_baseline
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc


def compare(baselines, resroot='results',
            outpath='figures/compare_baselines.png',
            end_year=2125):
    fig, ax = plt.subplots(figsize=(12, 6))
    base_colors = sc.gridcolors(len(baselines))

    rows = []
    for bi, base in enumerate(baselines):
        csv_path = f'{resroot}/{base}/fig2_res_stats.csv'
        if not os.path.exists(csv_path):
            print(f'  [skip] {csv_path} missing')
            continue
        df = pd.read_csv(csv_path)
        for scen in df.scenario.unique():
            sub = df[df.scenario == scen].sort_values('year')
            ax.plot(sub.year, sub.cum_med, color=base_colors[bi],
                    linestyle=_linestyle(scen), label=f'{base} — {scen}' if bi == 0 or True else None)
            rows.append({
                'baseline': base,
                'scenario': scen,
                'cum_cancers_2125': float(sub.iloc[-1].cum_med),
                'cum_lb_2125': float(sub.iloc[-1].cum_lb),
                'cum_ub_2125': float(sub.iloc[-1].cum_ub),
            })

    ax.set_ylim(bottom=0)
    ax.set_title('Cumulative cancers in 2023/24 cohort — baseline comparison')
    sc.SIticks(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=100)

    out = pd.DataFrame(rows)
    if not out.empty:
        print('\n=== Cumulative cancers at 2125 (med) ===')
        print(out.pivot(index='scenario', columns='baseline',
                        values='cum_cancers_2125').to_string(float_format='%.0f'))
    print(f'\nSaved figure: {outpath}')


def _linestyle(scen):
    return {'No vaccination': '-',
            'Double dose': '--',
            'Single dose shipments': '-.',
            'Single dose actual': ':'}.get(scen, '-')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', nargs='+', required=True)
    parser.add_argument('--resroot', default='results')
    parser.add_argument('--outpath', default='figures/compare_baselines.png')
    args = parser.parse_args()

    compare(args.baselines, resroot=args.resroot, outpath=args.outpath)
    print('\nDone.')
