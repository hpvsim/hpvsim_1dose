'''
Utilities for multicalibration
'''

# Imports
import sciris as sc
import pandas as pd
import numpy as np
import locations as loc
from scipy.stats import norm, lognorm


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


def make_sb_data(location=None, dist_type='lognormal'):

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
        f=dict(dist=distf, par1=par1f, par2=par2f),
        m=dict(dist=distm, par1=par1m, par2=par2m),
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
    cal.sim_results = sc.objdict()
    for skey in calib.extra_sim_result_keys:
        cal.sim_results[skey] =
    cal.target_data = calib.target_data
    cal.df = calib.df.iloc[0:n_results, ]
    return cal

