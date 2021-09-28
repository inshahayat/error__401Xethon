import app1
import app2
import app3
import streamlit as st
PAGES = {
    "PRODUCT PERFORMANCE CHECKER": app1,
    "CUSTOMER TRACKING": app2,
    "PRODUCT RECOMENDATION SENDER": app3
}
st.sidebar.title('USE CASE')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()