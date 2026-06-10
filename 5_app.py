import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('best_model.pkl')

st.title('🌾 Predikcija prinosa pirinča')
st.write('Unesite vrednosti parametara da biste dobili procenu prinosa.')

col1, col2 = st.columns(2)

with col1:
    avg_rain = st.slider('Prosečna kiša (mm)', 0, 370, 60)
    Nitrogen = st.slider('Nitrogen (kg)', 0, 325000, 50000)
    POTASH = st.slider('Potash (kg)', 0, 68000, 10000)
    PHOSPHATE = st.slider('Phosphate (kg)', 0, 138000, 20000)
    INCEPTISOLS = st.slider('Inceptisols', 0.0, 1.0, 0.0)
    LOAMY_ALFISOL = st.slider('Loamy Alfisol', 0.0, 1.0, 0.0)
    ORTHIDS = st.slider('Orthids', 0.0, 1.0, 0.0)

with col2:
    PSAMMENTS = st.slider('Psamments', 0.0, 1.0, 0.0)
    SANDY_ALFISOL = st.slider('Sandy Alfisol', 0.0, 1.0, 0.0)
    UDOLLS_UDALFS = st.slider('Udolls Udalfs', 0.0, 1.0, 0.0)
    UDUPTS_UDALFS = st.slider('Udupts Udalfs', 0.0, 1.0, 0.0)
    USTALF_USTOLLS = st.slider('Ustalf Ustolls', 0.0, 1.0, 0.0)
    VERTIC_SOILS = st.slider('Vertic Soils', 0.0, 1.0, 0.0)
    VERTISOLS = st.slider('Vertisols', 0.0, 1.0, 0.0)

input_data = np.array([[
    avg_rain, Nitrogen, POTASH, PHOSPHATE,
    INCEPTISOLS, LOAMY_ALFISOL, ORTHIDS, PSAMMENTS,
    SANDY_ALFISOL, UDOLLS_UDALFS, UDUPTS_UDALFS,
    USTALF_USTOLLS, VERTIC_SOILS, VERTISOLS
]])

if st.button('Predvidi prinos'):
    prediction = model.predict(input_data)
    st.success(f'🌾 Procenjeni prinos: **{prediction[0]:.2f} kg/ha**')