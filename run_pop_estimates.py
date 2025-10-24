"""
Pull UN demographic data
"""

import numpy as np
import pandas as pd
import sciris as sc
import requests


def get_indicators(base_url, auth_key, do_save=True):
    return get_available(base_url, auth_key, 'indicators/', do_save=do_save)


def get_locations(base_url, auth_key, do_save=True):
    return get_available(base_url, auth_key, 'locations/', do_save=do_save)


def get_data(base_url, auth_key, which, do_save=False):
    df = get_available(base_url, auth_key, which, filename='mortality', do_save=do_save)
    return df


def get_available(base_url, auth_key, which, filename=None, do_save=True):
    """ Save a list of indicators or locations to a csv file """
    first_url = base_url + which
    payload = {}
    headers = {'Authorization': auth_key}
    response = requests.request("GET", first_url, headers=headers, data=payload)
    j = response.json()

    # Set up list of dataframes
    dfs = sc.autolist()
    dfs += pd.json_normalize(j['data'])

    while j['nextPage'] != None:
        pageno = j['pageNumber']
        print(f'Processing page {pageno}')
        new_target = j['nextPage'].split('?')[1]
        new_url = base_url + which + '/?' + new_target
        response = requests.request("GET", new_url, headers=headers, data=payload)
        j = response.json()
        df_temp = pd.json_normalize(j['data'])
        dfs += df_temp

    df = pd.concat(dfs)
    if do_save:
        if filename is None: filename = which.strip('/')
        df.to_csv(filename+'.csv')

    return df


if __name__ == '__main__':

    # Set up
    base_url = "https://population.un.org/dataportalapi/api/v1/"
    auth_key = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6InJvYnluYS5zQGdtYWlsLmNvbSIsIm5iZiI6MTc0MjgyNjQ0MiwiZXhwIjoxNzc0MzYyNDQyLCJpYXQiOjE3NDI4MjY0NDIsImlzcyI6ImRvdG5ldC11c2VyLWp3dHMiLCJhdWQiOiJkYXRhLXBvcnRhbC1hcGkifQ.EpRT56NPSKHaY5Bv-vBuN77zlkePQUso4TFXAPldnuU"

    to_run = [
        # 'age_dist',
        'collate',
    ]

    indicator_codes = dict(
        age_dist=47,  # Population size by age
    )

    country_codes = dict(
        bangladesh=50,  # Bangladesh
        burkina_faso=854,  # Burkina Faso
        cambodia=116,  # Cambodia
        cameroon=120,  # Cameroon
        cote_divoire=384,  # Cote d'Ivoire
        eritrea=232,  # Eritrea
        ethiopia=231,  # Ethiopia
        gambia=270,  # Gambia
        kenya=404,  # Kenya
        kosovo=312,  # Kosovo
        kyrgyzstan=417,  # Kyrgyzstan
        laos=418,  # Laos
        lesotho=426,  # Lesotho
        liberia=430,  # Liberia
        malawi=454,  # Malawi
        mali=466,  # Mali
        mauritania=478,  # Mauritania
        mozambique=508,  # Mozambique
        myanmar=104,  # Myanmar
        nepal=524,  # Nepal
        nigeria=566,  # Nigeria
        rwanda=646,  # Rwanda
        sao_tome_principe=678,  # Sao Tome and Principe
        senegal=686,  # Senegal
        sierra_leone=694,  # Sierra Leone
        solomon_islands=90,  # Solomon Islands
        sri_lanka=144,  # Sri Lanka
        tanzania=834,  # Tanzania
        togo=768,  # Togo
        uganda=800,  # Uganda
        zambia=894,  #
        zimbabwe=716,  # Zimbabwe
    )

    dfs = sc.autolist()

    if 'age_dist' in to_run:
        for location, country_code in country_codes.items():
            icode = indicator_codes['age_dist']
            start_year = 2023
            end_year = 2025
            target = f"data/indicators/{icode}/locations/{country_code}/start/{start_year}/end/{end_year}"
            df = get_data(base_url, auth_key, target, do_save=False)
            df = df.loc[(df.sex=='Female') & (df.variant=='Median') & (df.ageStart == 9)]
            label_dict = {'timeLabel': 'Time', 'ageStart': 'AgeGrpStart', 'value': 'PopTotal'}
            for ol, nl in label_dict.items():
                df = df.rename(columns={ol: nl})

            df = df[label_dict.values()]
            df['country'] = location
            df['id'] = country_code
            df.to_csv(f'{location}_age_data.csv')
            dfs += df

        big_df = pd.concat(dfs)
        big_df.to_csv(f'age_data.csv')

    if 'collate' in to_run:
        dfs = sc.autolist()
        for location in country_codes.keys():
            df = pd.read_csv(f'age_data/{location}_age_data.csv')
            dfs += df
        big_df = pd.concat(dfs)

        # Rearrange data so that year year is a column and the rows represent the countries
        new_df = pd.pivot_table(big_df, index=['country', 'id', 'AgeGrpStart'], columns='Time', values='PopTotal')

        new_df.to_csv(f'age_data.csv')

    print('Done.')