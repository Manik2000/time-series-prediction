import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from streamlit_utils import get_markdown_text


def home_page():
    st.markdown(get_markdown_text("home_page"))


def eda_page():
    st.markdown(get_markdown_text("eda_page"))


def arima_page():
    st.markdown(get_markdown_text("arima_page"))
    x = np.linspace(-2, 2, 500)
    y = x**2
    df = pd.DataFrame({'x': x, 'y': y})
    p = alt.Chart(df).mark_line().encode(x='x', y='y')
    st.altair_chart(p)


def lstm_page():
    st.markdown(get_markdown_text("lstm_page"))
