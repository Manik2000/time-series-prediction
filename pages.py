import os
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from streamlit_utils import get_markdown_text
from Climate import Country
from LSTM import LSTM
from Baseline import Baseline


def home_page():
    st.markdown(get_markdown_text("home_page"))


def eda_page():
    st.markdown(get_markdown_text("eda_page"))


def models_page():
    st.markdown(get_markdown_text('models_page'))

    path = os.path.join(os.getcwd(), "loss", "test", "country")
    models = ['LSTM', 'Baseline']  # to add later

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

    head_cols = st.columns([3, 1])

    option = head_cols[0].selectbox('Select Country', df.Country.unique())
    country = Country(option)
    head_cols[1].markdown("### Xi Correlation")
    head_cols[1].markdown(f"##### {round(country.correlation()['correlation'], 3)}")

    years = st.slider('Years', df.year.min(), 2100, (df.year.min(), df.year.max()))
    start, end = tuple(map(str, years))

    cols = st.columns(2)

    with cols[0]:
        model = st.selectbox('Select Model', ['Baseline', 'LSTM'])
        fig = Figure()
        st.plotly_chart(country.plot(fig, eval(model), start=start, end=end),
                        use_container_width=True)

    with cols[1]:
        order = st.slider("Regression Order", 1, 10, 3)
        fig = Figure()
        st.plotly_chart(country.plot(fig, eval(model), start=start, end=end, smoothed=True, order=order),
                        use_container_width=True)

    st.markdown("### Inflection Points")
    for inflection in country.inflection_points(order=order):
        st.markdown(f"##### {inflection.strftime('%b-%Y')}")


def map_page():  # TODO: temporary page

    st.markdown(get_markdown_text("map_page"))
    df = pd.read_csv('final_data.csv')
    df = df[df['year'] > 2000]

    fig = px.choropleth(df,
                        locations="iso_alpha",
                        color="AverageTemperature",
                        hover_name="Country",
                        animation_frame="dt",
                        range_color=(-30, 30),
                        )

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 15
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

    st.plotly_chart(fig, use_container_width=True)
