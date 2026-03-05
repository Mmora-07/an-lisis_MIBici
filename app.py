import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(
    page_title="Predicción Demanda MiBici",
    page_icon="🚲",
    layout="centered"
)

st.title("🚲 Predicción Inteligente de Demanda")
st.markdown("Simulación de cantidad de viajes por estación")

# ==========================================
# CARGAR RECURSOS
# ==========================================
@st.cache_resource
def cargar_recursos():
    modelo_dia = joblib.load("modelo_dia.pkl")
    modelo_hora = joblib.load("modelo_hora.pkl")

    with open("estaciones_dict.json", "r", encoding="utf-8") as f:
        estaciones_dict = json.load(f)

    estaciones_dict = {int(k): v for k, v in estaciones_dict.items()}

    return modelo_dia, modelo_hora, estaciones_dict

modelo_dia, modelo_hora, estaciones_dict = cargar_recursos()

# Obtener categorías reales del modelo (estaciones vistas en entrenamiento)
estaciones_validas = modelo_dia.booster_.pandas_categorical[0]

# Crear DataFrame estaciones
df_estaciones = pd.DataFrame.from_dict(estaciones_dict, orient="index")
df_estaciones["id"] = df_estaciones.index

# ==========================================
# SELECCIÓN ESTACIÓN
# ==========================================
st.subheader("Selecciona la estación")

modo_busqueda = st.radio(
    "Buscar estación por:",
    ["ID", "Nombre"],
    horizontal=True
)

if modo_busqueda == "ID":
    estacion_id = st.number_input("ID Estación", min_value=0, step=1)
    nombre_estacion = estaciones_dict.get(estacion_id, {}).get("name", "No encontrada")
    st.info(f"Nombre estación: {nombre_estacion}")
else:
    nombre_seleccionado = st.selectbox(
        "Nombre estación",
        sorted(df_estaciones["name"].unique())
    )
    estacion_id = int(df_estaciones[df_estaciones["name"] == nombre_seleccionado]["id"].values[0])
    st.info(f"ID estación: {estacion_id}")

# Validar estación contra entrenamiento
if estacion_id not in estaciones_validas:
    st.error("⚠️ Esta estación no fue utilizada durante el entrenamiento del modelo.")
    st.stop()

# ==========================================
# SELECCIÓN DÍA
# ==========================================
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

# ==========================================
# HORAS DISPONIBLES (evitar 1–4)
# ==========================================
horas_validas = [h for h in range(24) if h not in [1,2,3,4]]

hora = st.selectbox("Hora del día", horas_validas)

# ==========================================
# PREDICCIÓN
# ==========================================
if st.button("🔮 Generar Predicción"):

    # ============================
    # Predicción diaria
    # ============================
    input_dia = pd.DataFrame({
        "Origen_Id": [estacion_id],
        "dia_semana": [dia_semana]
    })

    input_dia["Origen_Id"] = pd.Categorical(
        input_dia["Origen_Id"],
        categories=estaciones_validas
    )

    input_dia["dia_semana"] = input_dia["dia_semana"].astype("category")

    pred_dia = modelo_dia.predict(input_dia)[0]

    st.success(f"📅 Demanda estimada del día: {round(pred_dia, 2)} viajes")

    # ============================
    # Predicción por hora
    # ============================
    input_hora = pd.DataFrame({
        "Origen_Id": [estacion_id],
        "dia_semana": [dia_semana],
        "hora": [hora]
    })

    input_hora["Origen_Id"] = pd.Categorical(
        input_hora["Origen_Id"],
        categories=estaciones_validas
    )

    input_hora["dia_semana"] = input_hora["dia_semana"].astype("category")

    pred_hora = modelo_hora.predict(input_hora)[0]

    st.success(f"⏰ Demanda estimada para esa hora: {round(pred_hora, 2)} viajes")

    # ============================
    # Curva 24h (con horas 1–4 = 0)
    # ============================
    horas = list(range(24))
    pred_24h = []

    for h in horas:
        if h in [1,2,3,4]:
            pred_24h.append(0)
        else:
            temp = pd.DataFrame({
                "Origen_Id": [estacion_id],
                "dia_semana": [dia_semana],
                "hora": [h]
            })

            temp["Origen_Id"] = pd.Categorical(
                temp["Origen_Id"],
                categories=estaciones_validas
            )
            temp["dia_semana"] = temp["dia_semana"].astype("category")

            pred_24h.append(modelo_hora.predict(temp)[0])

    # ============================
    # Gráfico
    # ============================
    st.subheader("📊 Demanda estimada por hora")

    fig, ax = plt.subplots()
    ax.plot(horas, pred_24h)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Cantidad estimada de viajes")
    ax.set_title("Curva de demanda horaria estimada")
    ax.grid(True)

    st.pyplot(fig)