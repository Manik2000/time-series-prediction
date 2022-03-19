import streamlit as st
from PIL import Image, ImageOps
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
    "Map": map_page,
}


def show_page(page):
    st.session_state.page = page    
    

def show_app():
    st.sidebar.subheader("Menu")
    for t in mapping.keys():    
        st.sidebar.button(t, on_click=show_page, key='menu' + t, args=(t, ))
    st.sidebar.markdown(get_markdown_text("about_1"))
    image = Image.open('markdown/gauss_logo.png')
    new_image = Image.new("RGBA", image.size, 'WHITE')
    image.paste(new_image, (0, 0), image)
    st.sidebar.image(image)
    st.sidebar.markdown(get_markdown_text("about_2"))

    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    page = st.session_state.page
    func = mapping[page]    
    func()
    st.write("")


if __name__ == "__main__":
    show_app()
