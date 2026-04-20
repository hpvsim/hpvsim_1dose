'''
Utilities for multicalibration
'''

# Imports
import os

import sciris as sc
import pandas as pd
import numpy as np
import locations as loc
from scipy.stats import norm, lognorm


# ------ CSV extractors (run once from existing .obj/.mres → plot-ready CSVs) ------

SCENARIO_NAMES = ['No vaccination', 'Double dose',
                  'Single dose shipments', 'Single dose actual']


def extract_fig2_csvs(locations, vx_scens_dir='results', out_dir='results'):
    """Aggregate `raw_cohort_cancers` across countries × seeds × time.

    Emits:
      fig2_res_stats.csv   — scenario, year, cum_med, cum_lb, cum_ub
      fig2_diffs.csv       — comparison, year, cum_med, cum_lb, cum_ub
    """
    os.makedirs(out_dir, exist_ok=True)

    # raw_cohort_cancers covers the 2024+ cohort (102 years × 20 seeds per country)
    stacks = {s: None for s in SCENARIO_NAMES}
    for location in locations:
        mres = sc.loadobj(f'{vx_scens_dir}/{location.replace(" ", "_")}_vx_scens.obj')
        for sname in SCENARIO_NAMES:
            arr = np.asarray(mres[sname]['raw_cohort_cancers'])
            stacks[sname] = arr.copy() if stacks[sname] is None else stacks[sname] + arr
    n_years = stacks[SCENARIO_NAMES[0]].shape[0]
    year = np.arange(2024, 2024 + n_years)

    def _cum_quantiles(arr):
        return (np.cumsum(np.quantile(arr, 0.5, axis=-1)),
                np.cumsum(np.quantile(arr, 0.1, axis=-1)),
                np.cumsum(np.quantile(arr, 0.9, axis=-1)))

    ts_rows = []
    for sname, arr in stacks.items():
        med, lb, ub = _cum_quantiles(arr)
        for yi, yr in enumerate(year):
            ts_rows.append({'scenario': sname, 'year': float(yr),
                            'cum_med': float(med[yi]),
                            'cum_lb': float(lb[yi]),
                            'cum_ub': float(ub[yi])})
    pd.DataFrame(ts_rows).to_csv(f'{out_dir}/fig2_res_stats.csv',
                                 index=False, float_format='%.4f')

    diffs_rows = []
    for label, (a, b) in {
        'Single dose shipments vs Double dose': (stacks['Double dose'], stacks['Single dose shipments']),
        'Single dose actual vs Double dose':     (stacks['Double dose'], stacks['Single dose actual']),
    }.items():
        diff = a - b
        med = np.cumsum(np.median(diff, axis=1))
        lb = np.cumsum(np.quantile(diff, 0.1, axis=1))
        ub = np.cumsum(np.quantile(diff, 0.9, axis=1))
        for yi, yr in enumerate(year):
            diffs_rows.append({'comparison': label, 'year': float(yr),
                               'cum_med': float(med[yi]),
                               'cum_lb': float(lb[yi]),
                               'cum_ub': float(ub[yi])})
    pd.DataFrame(diffs_rows).to_csv(f'{out_dir}/fig2_diffs.csv',
                                    index=False, float_format='%.4f')


def extract_figS5_asr_csv(locations, mres_dir='raw_results', out_dir='results',
                          start_year=2000):
    """Per-country asr_cancer_incidence time series → figS5_asr.csv."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for location in locations:
        res = sc.loadobj(f'{mres_dir}/{location.replace(" ", "_")}.mres')
        years = np.asarray(res['year'])
        r = res['asr_cancer_incidence']
        mask = years >= start_year
        yy = years[mask]
        vv = np.asarray(r.values)[mask]
        lo = np.asarray(r.low)[mask]
        hi = np.asarray(r.high)[mask]
        for yi, yr in enumerate(yy):
            rows.append({'location': location, 'year': float(yr),
                         'value': float(vv[yi]),
                         'low': float(lo[yi]),
                         'high': float(hi[yi])})
    pd.DataFrame(rows).to_csv(f'{out_dir}/figS5_asr.csv',
                              index=False, float_format='%.4f')


def extract_figS6_country_ts_csv(locations, vx_scens_dir='results', out_dir='results'):
    """Per-country per-scenario cumulative-cancer quantiles → figS6_country_ts.csv."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for location in locations:
        mres = sc.loadobj(f'{vx_scens_dir}/{location.replace(" ", "_")}_vx_scens.obj')
        years = np.arange(2024, 2126)
        for sname, scen_mres in mres.items():
            arr = np.asarray(scen_mres['raw_cohort_cancers'])
            med = np.cumsum(np.quantile(arr, 0.5, axis=-1))
            lb = np.cumsum(np.quantile(arr, 0.1, axis=-1))
            ub = np.cumsum(np.quantile(arr, 0.9, axis=-1))
            for yi, yr in enumerate(years):
                rows.append({'location': location, 'scenario': sname,
                             'year': int(yr),
                             'cum_med': float(med[yi]),
                             'cum_lb': float(lb[yi]),
                             'cum_ub': float(ub[yi])})
    pd.DataFrame(rows).to_csv(f'{out_dir}/figS6_country_ts.csv',
                              index=False, float_format='%.4f')


def extract_figS4_calib_csvs(locations, calib_dir='results', out_dir='results',
                             resname='cancers', date=2020, filestem=''):
    """Per-country calibration trials by age bin → figS4_calib.csv + figS4_targets.csv.

    model CSV: location, bin_idx, bin_label, run_idx, value
    target CSV: location, bin_idx, bin_label, value
    """
    os.makedirs(out_dir, exist_ok=True)
    model_rows, target_rows = [], []
    for location in locations:
        dfl = location.replace(' ', '_')
        calib = sc.loadobj(f'{calib_dir}/{dfl}_calib{filestem}_reduced.obj')

        baseres = calib.analyzer_results[0][resname]
        bins = np.asarray(baseres['bins'])
        age_labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins) - 1)]
        age_labels.append(f'{int(bins[-1])}+')

        for run_idx, run in enumerate(calib.analyzer_results):
            vals = np.asarray(run[resname][date])
            for bi, v in enumerate(vals):
                model_rows.append({'location': location, 'bin_idx': bi,
                                   'bin_label': age_labels[bi] if bi < len(age_labels) else str(bi),
                                   'run_idx': run_idx, 'value': float(v)})

        target_df = calib.target_data[0]
        target_df = target_df[target_df.name == resname]
        for bi, v in enumerate(target_df.value.values):
            target_rows.append({'location': location, 'bin_idx': bi,
                                'bin_label': age_labels[bi] if bi < len(age_labels) else str(bi),
                                'value': float(v)})

    pd.DataFrame(model_rows).to_csv(f'{out_dir}/figS4_calib.csv',
                                    index=False, float_format='%.4f')
    pd.DataFrame(target_rows).to_csv(f'{out_dir}/figS4_targets.csv',
                                     index=False, float_format='%.4f')


def extract_all_csvs(locations=None, results_dir='results',
                     raw_results_dir='raw_results'):
    """Convenience: regenerate every plot-ready CSV from obj/mres."""
    if locations is None:
        locations = loc.locations
    extract_fig2_csvs(locations, vx_scens_dir=results_dir, out_dir=results_dir)
    extract_figS5_asr_csv(locations, mres_dir=raw_results_dir, out_dir=results_dir)
    extract_figS6_country_ts_csv(locations, vx_scens_dir=results_dir, out_dir=results_dir)
    extract_figS4_calib_csvs(locations, calib_dir=results_dir, out_dir=results_dir)
    print(f'Wrote fig2/figS4/figS5/figS6 CSVs to {results_dir}/')


def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


def map_sb_loc(location):
    ''' Map between different representations of country names '''
    location = location.title()
    if location == "Cote Divoire": location = "Cote d'Ivoire"
    return location


def rev_map_sb_loc(location):
    ''' Map between different representations of country names '''
    location = location.lower()
    # location = location.replace(' ', '_')
    if location == "cote d'ivoire": location = 'cote divoire'
    return location


def make_sb_data(location=None, dist_type='lognormal', debut_bias=None):

    if debut_bias is None:
        debut_bias = [0,0]

    # Deal with missing countries and different spelling conventions
    if location in loc.nosbdata_locations:
        sb_location = 'Ethiopia'   # Use assumptions for Ethiopia for CDI
    else:
        sb_location = map_sb_loc(location)

    # Read in data
    sb_data_f = pd.read_csv(f'data/sb_pars_women_{dist_type}.csv')
    sb_data_m = pd.read_csv(f'data/sb_pars_men_{dist_type}.csv')

    try:
        distf = sb_data_f.loc[sb_data_f["location"]==sb_location,"dist"].iloc[0]
        par1f = sb_data_f.loc[sb_data_f["location"]==sb_location,"par1"].iloc[0]
        par2f = sb_data_f.loc[sb_data_f["location"]==sb_location,"par2"].iloc[0]
        distm = sb_data_m.loc[sb_data_m["location"]==sb_location,"dist"].iloc[0]
        par1m = sb_data_m.loc[sb_data_m["location"]==sb_location,"par1"].iloc[0]
        par2m = sb_data_m.loc[sb_data_m["location"]==sb_location,"par2"].iloc[0]
    except:
        print(f'No data for {sb_location=}, {location=}')

    debut = dict(
        f=dict(dist=distf, par1=par1f+debut_bias[0], par2=par2f),
        m=dict(dist=distm, par1=par1m+debut_bias[1], par2=par2m),
    )

    return debut


def make_datafiles(locations):
    ''' Get the relevant datafiles for the selected locations '''
    datafiles = dict()
    cancer_type_locs = ['ethiopia', 'mozambique', 'nigeria', 'tanzania', 'uganda']

    for location in locations:
        dflocation = location.replace(' ', '_')
        datafiles[location] = [
            f'data/{dflocation}_cancer_cases.csv',
            f'data/{dflocation}_asr_cancer_incidence.csv',
        ]

        if location in cancer_type_locs:
            datafiles[location] += [f'data/{dflocation}_cancer_types.csv']

    return datafiles


def read_debut_data(dist_type='lognormal'):
    '''
    Read in dataframes taken from DHS and return them in a plot-friendly format,
    optionally saving the distribution parameters
    '''

    df1 = pd.read_csv('data/afs_dist.csv')
    df2 = pd.read_csv('data/afs_median.csv')

    # Deal with median data
    df2['y'] = 50

    # Rearrange data into a plot-friendly format
    dff = {}
    rvs = {'Women': {}, 'Men': {}}

    for sex in ['Women', 'Men']:

        dfw = df1[['Country', f'{sex} 15', f'{sex} 18', f'{sex} 20', f'{sex} 22', f'{sex} 25', f'{sex} never']]
        dfw = dfw.melt(id_vars='Country', value_name='Percentage', var_name='AgeStr')

        # Add values for proportion ever having sex
        countries = dfw.Country.unique()
        n_countries = len(countries)
        vals = []
        for country in countries:
            val = 100-dfw.loc[(dfw['AgeStr'] == f'{sex} never') & (dfw['Country'] == country) , 'Percentage'].iloc[0]
            vals.append(val)

        data_cat = {'Country': countries, 'AgeStr': [f'{sex} 60']*n_countries}
        data_cat["Percentage"] = vals
        df_cat = pd.DataFrame.from_dict(data_cat)
        dfw = pd.concat([dfw,df_cat])

        conditions = [
            (dfw['AgeStr'] == f"{sex} 15"),
            (dfw['AgeStr'] == f"{sex} 18"),
            (dfw['AgeStr'] == f"{sex} 20"),
            (dfw['AgeStr'] == f"{sex} 22"),
            (dfw['AgeStr'] == f"{sex} 25"),
            (dfw['AgeStr'] == f"{sex} 60"),
        ]
        values = [15, 18, 20, 22, 25, 60]
        dfw['Age'] = np.select(conditions, values)

        dff[sex] = dfw

        res = dict()
        res["location"] = []
        res["par1"] = []
        res["par2"] = []
        res["dist"] = []
        for pn,country in enumerate(countries):
            dfplot = dfw.loc[(dfw["Country"] == country) & (dfw["AgeStr"] != f'{sex} never') & (dfw["AgeStr"] != f'{sex} 60')]
            x1 = 15
            p1 = dfplot.loc[dfplot["Age"] == x1, 'Percentage'].iloc[0] / 100
            x2 = df2.loc[df2["Country"]==country,f"{sex} median"].iloc[0]
            p2 = .50
            # x2 = 25
            # p2 = dfplot.loc[dfplot["Age"] == x2, 'Percentage'].iloc[0] / 100
            res["location"].append(country)
            res["dist"].append(dist_type)

            s, scale = logn_percentiles_to_pars(x1, p1, x2, p2)
            rv = lognorm(s=s, scale=scale)
            res["par1"].append(rv.mean())
            res["par2"].append(rv.std())

            rvs[sex][country] = rv

        pd.DataFrame.from_dict(res).to_csv(f'data/sb_pars_{sex.lower()}_{dist_type}.csv')

    return countries, dff, df2, rvs


def logn_percentiles_to_pars(x1, p1, x2, p2):
    """ Find the parameters of a lognormal distribution where:
            P(X < p1) = x1
            P(X < p2) = x2
    """
    x1 = np.log(x1)
    x2 = np.log(x2)
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)
    s = (x2 - x1) / (p2ppf - p1ppf)
    mean = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = np.exp(mean)
    return s, scale


def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale


def shrink_calib(calib, n_results=100):
    cal = sc.objdict()
    plot_indices = calib.df.iloc[0:n_results, 0].values
    cal.analyzer_results = [calib.analyzer_results[i] for i in plot_indices]
    cal.target_data = calib.target_data
    cal.df = calib.df.iloc[0:n_results, ]
    return cal

