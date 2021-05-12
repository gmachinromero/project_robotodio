
# ------------------------------------------------------------------------------
# Python libraries
# ------------------------------------------------------------------------------

# Twitter API
import tweepy
import os

import streamlit as st


# ------------------------------------------------------------------------------
# Streamlit appereance
# ------------------------------------------------------------------------------

# Appeareance
st.set_page_config(
    page_title = "robotodio",
    page_icon = "👁",
    layout = "wide",
    initial_sidebar_state = "expanded"
    )

# Title
with st.beta_container():
    st.write("""
    # Robotodio 👁 🔪
    *Analizando la agresividad en Twitter*
    """)
    st.markdown("---")

# Sidebar
st.sidebar.title('Introduce una cuenta de Twitter')
target = st.sidebar.text_input("sin @")

# Reloj
with st.spinner('Tened paciencia y tendréis ciencia...⏳'):

    st.write(os.environ["db_username"])

    st.write(os.environ["prueba_secreto"])

# Reloj
st.success('Terminado!')