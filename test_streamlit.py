import streamlit as st

st.set_page_config(
    page_title="ResearchLens Test",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ResearchLens")
st.write("Streamlit UI is working.")

st.success("If you can see this, Streamlit itself is working.")

question = st.text_input("Test question")

if question:
    st.write("You entered:", question)
