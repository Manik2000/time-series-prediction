import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from plotly.graph_objects import Figure
from streamlit_utils import get_markdown_text
from Climate import Country


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


def analysis_page():
    st.markdown(get_markdown_text('analysis_page'))

    df = pd.read_csv('final_data.csv')

    head_cols = st.columns([3, 1])

    option = head_cols[0].selectbox('Select Country',
                          df.Country.unique())
    country = Country(option)
    head_cols[1].markdown("### Correlation")
    head_cols[1].markdown(f"##### {round(country.correlation()['correlation'], 3)}")

    start = st.slider('Start Year', df.year.min(), df.year.max()-1)
    end = st.slider('End Year', start+1, df.year.max(), df.year.max())
    start, end = str(start), str(end)

    cols = st.columns(2)

    with cols[0]:
        st.markdown("#")
        st.markdown("##")
        st.markdown("##")
        fig = Figure()
        st.plotly_chart(country.plot(fig, start=start, end=end), use_container_width=True)

    with cols[1]:
        order = st.slider("Regression Order", 1, 10, 3)
        fig = Figure()
        st.plotly_chart(country.plot(fig, start=start, end=end, smoothed=True, order=order), use_container_width=True)

    st.markdown("### Inflection Points")
    for inflection in country.inflection_points(start=start, end=end, order=order):
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
