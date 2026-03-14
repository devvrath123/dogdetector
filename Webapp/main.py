import streamlit as st
from model import load_model, create_transforms, predict, get_breeds, get_breeds_dict, update_breeds
import time

@st.cache_resource
def cache_model():
    return load_model('Models/dog-classifier-ConvNextSmall-webapp_merged-classes.pth', 119)

@st.cache_resource
def cache_breeds():
    return get_breeds('breeds.csv')

def homepage():
    st.title('🐶 Welcome to Dog Detector!')
    st.divider()
    st.write(
        'Dog Detector is a neural network powered web-app that can detect your dog\'s breed with just a picture. Try it out below by uploading an image!'
    )
    with st.container(horizontal=True):
        st.write("Check out the **FAQ** and **Metrics** pages:")
        st.page_link(st.Page("faq.py", title="FAQ", icon="📝"))
        st.page_link(st.Page("stats.py", title="Prediction Metrics", icon="📊"))
        
    breeds_dict = get_breeds_dict('breeds.csv')

    col1, col2 = st.columns(2, gap="large")
    with col1:
        dog = st.file_uploader('Upload your image', type=["jpg", "jpeg", "png", "webp", "avif", "jfif"])
        st.caption("Accepted file formats: **JPG, JPEG, PNG, WEBP, AVIF, JFIF**")
    if dog:
        with col2:
            model = cache_model()
            breeds = cache_breeds()
            with st.container(horizontal_alignment='center'):
                st.image(dog)
        st.divider()
        pbar = st.progress(0, text="Thinking..")
        for percent in range(100):
            time.sleep(0.005)
            pbar.progress(percent+1, text="Thinking..")
        prediction, prob = predict(dog, model, breeds, create_transforms())
        st.session_state['last_prediction'] = prediction
        result = ""
        pbar.empty()
        if prediction.startswith(('A', 'E', 'I', 'O', 'U')):
            result = f"That's an {prediction}!"
        else:
            result = f"That's a {prediction}!"

        if prob < 40:
            st.markdown(
                f"""
                <div style="
                    background-color: #f8d7da; 
                    color: #721c24; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid #dc3545;
                    text-align: center;">
                    <h2 style="margin: 0;">Hmmm, that doesn't look like a dog.</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.balloons()
            st.markdown(
                f"""
                <div style="
                    background-color: #d4edda; 
                    color: #155724; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid #28a745;
                    text-align: center;">
                    <h2 style="margin: 0;">{result}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
            breeds_dict[prediction] += 1
            update_breeds('breeds.csv', breeds_dict)


pg = st.navigation([
    st.Page(homepage, title="Home", icon="🐶"),
    st.Page("faq.py", title="FAQ", icon="📝"),
    st.Page("stats.py", title="Prediction Metrics", icon="📊"),
    st.Page("about.py", title="Contact", icon="👤")
])
pg.run()