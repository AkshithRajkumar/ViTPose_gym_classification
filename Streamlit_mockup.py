# Streamlit file with file uploader and video player
import streamlit as st
import base64

st.set_page_config(page_title='Welcome',
                   page_icon="👋")


st.title('ViTPose Gym Classification')

st.write('This is a web application that uses the ViTPose model to classify gym exercises. Upload a video of a gym exercise to get started!')

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write('File uploaded successfully!')

    # play the video
    st.video(uploaded_file)


