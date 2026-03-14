import streamlit as st

st.title("👤 Contact Info")
st.divider()
st.write("This web-app was created by Devvrath S. Find contact links below:")
with st.container(horizontal=True):
    st.link_button(label="GitHub", url="https://github.com/devvrath123")
    st.link_button(label="LinkedIn", url="https://www.linkedin.com/in/devvrath-s-18b77b257/")