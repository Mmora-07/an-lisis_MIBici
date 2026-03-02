import streamlit as st
import pandas as pd
import joblib

# ====================================
# CONFIGURACIÓN
# ====================================
st.set_page_config(page_title="Predicción Demanda MiBici", layout="centered")

st.title("🚲 Predicción de Demanda de Viajes")
st.markdown("Modelo predictivo de cantidad de viajes por estación")

# ====================================
# CARGAR MODELOS
# ====================================
@st.cache_resource
def cargar_modelos():
    modelo_dia = joblib.load("modelo_dia.pkl")
    modelo_hora = joblib.load("modelo_hora.pkl")
    return modelo_dia, modelo_hora

modelo_dia, modelo_hora = cargar_modelos()

# ====================================
# SELECCIÓN DE MODELO
# ====================================
tipo_modelo = st.radio(
    "Selecciona tipo de predicción:",
    ["📅 Predicción por Día", "⏰ Predicción por Día y Hora"]
)

# ====================================
# INPUTS
# ====================================
estacion = st.number_input("ID Estación de Origen", min_value=0, step=1)

dias_dict = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo"
}

dia_semana = st.selectbox(
    "Día de la semana",
    options=list(dias_dict.keys()),
    format_func=lambda x: dias_dict[x]
)

if tipo_modelo == "⏰ Predicción por Día y Hora":
    hora = st.slider("Hora del día", 0, 23, 8)

# ====================================
# PREDICCIÓN
# ====================================
if st.button("Predecir demanda"):

    try:

        if tipo_modelo == "📅 Predicción por Día":

            input_data = pd.DataFrame({
                "Origen_Id": [estacion],
                "dia_semana": [dia_semana]
            })

            input_data["Origen_Id"] = input_data["Origen_Id"].astype("category")
            input_data["dia_semana"] = input_data["dia_semana"].astype("category")

            pred = modelo_dia.predict(input_data)[0]

        else:

            input_data = pd.DataFrame({
                "Origen_Id": [estacion],
                "dia_semana": [dia_semana],
                "hora": [hora]
            })

            input_data["Origen_Id"] = input_data["Origen_Id"].astype("category")
            input_data["dia_semana"] = input_data["dia_semana"].astype("category")

            pred = modelo_hora.predict(input_data)[0]

        st.success(f"📊 Cantidad estimada de viajes: {round(pred, 2)}")

    except Exception as e:
        st.error("Ocurrió un error en la predicción.")
        st.write(e)