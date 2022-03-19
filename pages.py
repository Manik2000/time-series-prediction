import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from streamlit_utils import get_markdown_text
from Climate import Country
from LSTM import LSTM
from Baseline import Baseline
from Arima import Arima
from boosting import XGBoost
from datetime import date


def home_page():
    st.markdown(get_markdown_text("home_page"))


def eda_page():
    st.markdown(get_markdown_text("eda_page"))


def models_page():
    st.markdown(get_markdown_text('models_page'))

    path = os.path.join(os.getcwd(), "loss", "test", "country")
    models = ['LSTM', 'Arima', 'XGBoost']

    def read_loss(model):
        df = pd.read_csv(os.path.join(path, f'{model}.csv'))
        df['Model'] = model

        return df

    merge_loss = lambda models: pd.concat([read_loss(model) for model in models])

    df = merge_loss(models)

    st.plotly_chart(px.box(df, x='Model', y="Loss", color="Continent"),
                    use_container_width=True)


def analysis_page():
    st.markdown(get_markdown_text('analysis_page'))

    df = pd.read_csv('final_data.csv')
    df = df[~df.is_continent]

    head_cols = st.columns([3, 1])

    option = head_cols[0].selectbox('Select Country', df.Country.unique())
    country = Country(option)
    head_cols[1].markdown("### Xi Correlation")
    head_cols[1].markdown(f"##### {round(country.correlation()['correlation'], 3)}")

    years = st.slider('Years', df.year.min(), 2100, (df.year.min(), df.year.max()))
    start, end = tuple(map(str, years))

    cols = st.columns([5, 5, 1])

    cols[2].markdown(f"##### ")
    lowess = cols[2].checkbox('LOWESS')

    with cols[0]:
        model = st.selectbox('Select Model', ['Baseline', 'LSTM', 'Arima', 'XGBoost'])
        fig = Figure()
        st.plotly_chart(country.plot(fig, eval(model), start=start, end=end),
                        use_container_width=True)

    with cols[1]:
        order = st.slider("Regression Order", 1, 10, 3)
        fig = Figure()
        if lowess:
            st.plotly_chart(country.plot(fig, eval(model), start=start, end=end, smoothed=True, level=order/10),
                            use_container_width=True)
        else:
            st.plotly_chart(country.plot(fig, eval(model), start=start, end=end, smoothed=True, order=order),
                            use_container_width=True)

    st.markdown("### Inflection Points")
    if lowess:
        for inflection in country.inflection_points(level=order/10):
            st.markdown(f"##### {inflection.strftime('%b-%Y')}")
    else:
        for inflection in country.inflection_points(order=order):
            st.markdown(f"##### {inflection.strftime('%b-%Y')}")


def map_page():

    st.markdown(get_markdown_text("map_page"))

    df = pd.read_csv('final_data.csv')
    df = df[~df.is_continent]

    years = st.slider('Years', df.year.min(), 2100, (1860, df.year.max()))
    start, end = years

    df = df[np.logical_and(start <= df['year'], df['year'] <= end)]

    cols = st.columns(2)

    model = cols[0].selectbox('Select Model', ['Baseline', 'LSTM', 'Arima', 'XGBoost'])
    smoothing = cols[1].slider('Yearly Smoothing', 1, 10, 1)

    for country in df.Country.unique()[:10]:
        country_model = Country(country)
        horizon = int((end - df[
            df['Country'] == country].year.max()) * 12 + 12 - pd.to_datetime(df[df['Country'] == country].dt.max()).month)
        preds = country_model.predict(eval(model), horizon)
        df = pd.concat([df, preds])

    df['smooth'] = df['year'] // smoothing * smoothing
    smoothed_df = df.groupby(['Country', 'smooth'])['AverageTemperature'].mean().reset_index()
    df.drop_duplicates(subset=['smooth', 'Country'], inplace=True)
    smoothed_df = pd.merge(smoothed_df, df, on=['smooth', 'Country'], suffixes=['', '_']).sort_values('smooth')

    fig = px.choropleth(smoothed_df,
                        locations="iso_alpha",
                        color="AverageTemperature",
                        hover_name="Country",
                        animation_frame="smooth",
                        range_color=(-30, 30))

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100 * smoothing
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 100 * smoothing

    st.plotly_chart(fig, use_container_width=True)
