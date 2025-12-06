import streamlit as st

st.set_page_config(page_title="Test App", layout="centered")

st.title("Streamlit Test Page")
st.write("If you can see this, your local Streamlit setup is working!")

name = st.text_input("Your name:")
if name:
    st.success(f"Hello, {name}! Streamlit is running locally")

if st.button("Test Button"):
    st.info("Button clicked!")