import streamlit as st
from pages import models_page, eda_page, home_page, analysis_page, map_page
from streamlit_utils import get_markdown_text


st.set_page_config(layout="wide")

st.markdown('''
<style>
button {
    width: 100% !important
}
</style>''', unsafe_allow_html=True)


mapping = {
    "Home": home_page,
    "EDA": eda_page,
    "Models": models_page,
    "Analysis": analysis_page,
    "Map tests (temporary page)": map_page,
}


def show_page(page):
    st.session_state.page = page    
    

def show_app():
    st.sidebar.subheader("Menu")
    for t in mapping.keys():    
        st.sidebar.button(t, on_click=show_page, key='menu' + t, args=(t, ))
    st.sidebar.markdown(get_markdown_text("about"))

    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    page = st.session_state.page
    func = mapping[page]    
    func()
    st.write("")


if __name__ == "__main__":
    show_app()
