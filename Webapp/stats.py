import streamlit as st
import pandas as pd
from model import get_breeds_dict

st.title('📊 Prediction Metrics')
st.divider()
st.write(
    'Find some interesting stats about recent predictions here:'
)

breeds_dict = get_breeds_dict('breeds.csv')
last_prediction = st.session_state.get('last_prediction', None)

popular = max(breeds_dict, key=breeds_dict.get)
count = breeds_dict[popular]
if count > 0:
    popular_breed = popular
    if last_prediction and (count == breeds_dict[last_prediction]):
        popular_breed = last_prediction
        count = breeds_dict[last_prediction]
    delta = ""
    if count == 1:
        delta += str(count) + ' prediction (all time)'
    else:
        delta += str(count) + ' predictions (all time)'
    st.metric("Most Popular Breed", popular_breed, delta=delta)
    breeds_df = pd.DataFrame(breeds_dict.items(), columns=['Breed', 'Predictions'])
    top10 = breeds_df.sort_values(by="Predictions", ascending=False).head(10).reset_index(drop=True)

    st.subheader('Top 10 Predictions Chart')
    st.bar_chart(top10, x='Breed', y='Predictions')
    st.caption('Metrics may include incorrect predictions', text_alignment="center")
else:
    st.write('No prediction data yet. Check back later.')