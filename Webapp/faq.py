import streamlit as st
import json

st.title("📝 Frequently Asked Questions")
st.divider()
st.write("Here you can find important information about the web-app:")

@st.cache_data
def load_qna():
    with open('qna.json', 'r') as f:
        qna_list = json.load(f)
    return qna_list

qna_list = load_qna()
for dict in qna_list:
    with st.expander(dict['q']):
        st.write(dict['a'])