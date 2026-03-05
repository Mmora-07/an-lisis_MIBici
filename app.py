import streamlit as st
import pandas as pd
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
st.markdown("Estimación de viajes por estación durante el día")

# ==========================================
# CARGAR RECURSOS
# ==========================================

@st.cache_resource
def cargar_recursos():

    modelo = joblib.load("modelo_hora.pkl")

    with open("estaciones_dict.json", "r", encoding="utf-8") as f:
        estaciones_dict = json.load(f)

    estaciones_dict = {int(k): v for k, v in estaciones_dict.items()}

    return modelo, estaciones_dict


modelo, estaciones_dict = cargar_recursos()

# estaciones usadas en entrenamiento
estaciones_validas = modelo.booster_.pandas_categorical[0]

# dataframe estaciones
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

    if nombre_estacion != "No encontrada":
        st.success(f"📍 {nombre_estacion}")
    else:
        st.warning("Estación no encontrada")

else:

    nombre_seleccionado = st.selectbox(
        "Nombre estación",
        sorted(df_estaciones["name"].unique())
    )

    estacion_id = int(
        df_estaciones[df_estaciones["name"] == nombre_seleccionado]["id"].values[0]
    )

    st.success(f"ID estación: {estacion_id}")

# validar estación
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
# PREDICCIÓN
# ==========================================

if st.button("🔮 Generar Predicción"):

    horas = list(range(24))
    pred_24h = []

    for h in horas:

        # sistema no opera entre 1 y 4
        if h in [1,2,3,4]:
            pred_24h.append(0)
            continue

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
        temp["hora"] = temp["hora"].astype("category")

        pred = modelo.predict(temp)[0]

        pred_24h.append(pred)

    # ======================================
    # RESULTADOS
    # ======================================

    pred_dia = sum(pred_24h)

    st.metric(
        label="🚲 Viajes estimados en el día",
        value=f"{round(pred_dia,2)}"
    )

    # dataframe para gráfico
    df_pred = pd.DataFrame({
        "Hora": horas,
        "Viajes_estimados": pred_24h
    })

    # ======================================
    # GRÁFICO
    # ======================================

    st.subheader("📊 Demanda estimada por hora")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        df_pred["Hora"],
        df_pred["Viajes_estimados"],
        marker="o",
        linewidth=2
    )

    ax.fill_between(
        df_pred["Hora"],
        df_pred["Viajes_estimados"],
        alpha=0.2
    )

    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Cantidad estimada de viajes")
    ax.set_title("Curva de demanda diaria")

    ax.set_xticks(range(24))
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

    # ======================================
    # TABLA
    # ======================================

    st.subheader("Detalle por hora")
    st.dataframe(df_pred, use_container_width=True)