import pycountry_convert as pc
import pandas as pd


def create_continents(df):

    def get_continent(country):

        try:
            return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
        except:
            pass

    df['Continent'] = df.apply(lambda row: get_continent(row.Country), axis=1)

    return df


def preprocess(in_file='final_data.csv', out_file='final_data.csv'):

    df = pd.read_csv(in_file)
    df = create_continents(df)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':

    preprocess()
