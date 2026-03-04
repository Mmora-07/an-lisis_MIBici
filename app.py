import streamlit as st
import pandas as pd
import joblib
import json

modelo_dia = joblib.load("modelo_dia.pkl")

with open("data/estaciones_dict.json", "r", encoding="utf-8") as f:
    estaciones_dict = json.load(f)

estaciones_dict = {int(k): v for k, v in estaciones_dict.items()}

st.set_page_config(layout="wide")

# -----------------------
# Cargar modelo y datos
# -----------------------
modelo_dia = joblib.load("modelo_dia.pkl")

# Obtener estaciones entrenadas
estaciones_validas = list(modelo_dia.feature_name_)
# Mejor:
estaciones_validas = modelo_dia.booster_.pandas_categorical[0]

# -----------------------
# UI
# -----------------------
st.title("🚲 Predicción Semanal de Viajes por Estación")

estacion_id = st.number_input("Ingrese ID de estación", min_value=0)

if estacion_id not in estaciones_validas:
    st.error("⚠ Esta estación no fue usada en el entrenamiento del modelo.")
else:

    nombre = estaciones_dict.get(estacion_id, {}).get("name", "Desconocida")
    st.success(f"Estación seleccionada: {nombre}")

    dias = [
        "Lunes", "Martes", "Miércoles",
        "Jueves", "Viernes", "Sábado", "Domingo"
    ]

    df_pred = pd.DataFrame({
        "Origen_Id": [estacion_id]*7,
        "dia_semana": dias
    })

    df_pred["Origen_Id"] = df_pred["Origen_Id"].astype("category")
    df_pred["dia_semana"] = df_pred["dia_semana"].astype("category")

    predicciones = modelo_dia.predict(df_pred)

    df_pred["Predicción"] = predicciones

    # -----------------------
    # Métricas resumen
    # -----------------------
    total_semana = int(df_pred["Predicción"].sum())
    dia_max = df_pred.loc[df_pred["Predicción"].idxmax(), "dia_semana"]
    dia_min = df_pred.loc[df_pred["Predicción"].idxmin(), "dia_semana"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total semanal estimado", total_semana)
    col2.metric("Día más demandado", dia_max)
    col3.metric("Día menos demandado", dia_min)

    # -----------------------
    # Gráfica
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(df_pred["dia_semana"], df_pred["Predicción"])

    ax.set_title(f"Demanda semanal - {nombre}", fontsize=14)
    ax.set_ylabel("Cantidad de viajes estimados")
    ax.set_xlabel("Día de la semana")
    ax.grid(axis="y", alpha=0.3)

    # Rotar etiquetas
    plt.xticks(rotation=45)

    st.pyplot(fig)