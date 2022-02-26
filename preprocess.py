import pycountry_convert as pc
import pandas as pd
import numpy as np
import re


CONTINENTS = ['Asia', 'Europe', 'South America', 'Africa', 'North America', 'Oceania']
MAPPING = {'AS': 'Asia', 'EU': 'Europe', 'SA': 'South America', 'AF': 'Africa', 'NA': 'North America', 'OC': 'Oceania'}


def no_colonies(df):

    colonies = df.loc[df.Country.str.contains(r'\([\w\s]*\)'), 'Country'].unique()

    maps = {}
    for colony in colonies:
        maps[re.sub(r'\s\([\w\s]*\)', '', colony)] = colony

    def change(df, name, full):
        df = df[df.Country != name]
        df.replace(full, name, inplace=True)

        return df

    for name, full in maps.items():
        df = change(df, name, full)

    return df


def create_continents(df):

    global CONTINENTS

    def get_continent(country):

        try:
            return MAPPING[pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))]
        except KeyError:
            if country in CONTINENTS:
                return country

    df['Continent'] = df.apply(lambda row: get_continent(row.Country), axis=1)

    return df


def interpolation(df):

    for country in df["Country"].unique():

        subset = df[df["Country"] == country]
        subset = subset.interpolate()
        df[df["Country"] == country] = subset

    return df


def preprocess(in_file="GlobalLandTemperaturesByCountry.csv", out_file='final_data.csv'):

    df = pd.read_csv(in_file)
    df['year'] = pd.DatetimeIndex(df['dt']).year
    df.drop(columns=['AverageTemperatureUncertainty'], inplace=True)
    df = no_colonies(df)
    df = create_continents(df)
    df = interpolation(df)
    df.dropna(inplace=True)
    df.to_csv(out_file, index=False)

    return df


if __name__ == '__main__':

    preprocess()
