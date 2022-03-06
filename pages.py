import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from streamlit_utils import get_markdown_text
import plotly.express as px


def home_page():
    st.markdown(get_markdown_text("home_page"))


def eda_page():
    st.markdown(get_markdown_text("eda_page"))


def arima_page():
    st.markdown(get_markdown_text("arima_page"))
    x = np.linspace(-2, 2, 500)
    y = x ** 2
    df = pd.DataFrame({'x': x, 'y': y})
    p = alt.Chart(df).mark_line().encode(x='x', y='y')
    st.altair_chart(p)


def xgboost_page():
    st.markdown(get_markdown_text("xgboost_page"))


def lstm_page():
    st.markdown(get_markdown_text("lstm_page"))


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
