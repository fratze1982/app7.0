import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(page_title="Flexible KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 Flexible KI-Vorhersage für Lackrezepturen")

uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

def robust_csv_reader(file, sep=';', decimal=','):
    bad_lines = []
    # Wir lesen die Datei manuell ein, zählen Spaltenanzahl in Header
    file.seek(0)
    first_line = file.readline().decode('utf-8')
    n_cols = len(first_line.strip().split(sep))
    file.seek(0)

    valid_lines = []
    for i, line in enumerate(file):
        decoded = line.decode('utf-8')
        cols = decoded.strip().split(sep)
        if len(cols) != n_cols:
            bad_lines.append((i+1, decoded.strip()))
        else:
            valid_lines.append(decoded)
    if bad_lines:
        st.warning(f"Folgende Zeilen haben abweichende Spaltenanzahl (erwartet {n_cols}):")
        for linenr, content in bad_lines:
            st.warning(f"Zeile {linenr}: {content}")

    # Aus den validen Zeilen einen String machen und mit pd.read_csv einlesen
    from io import StringIO
    data_str = first_line + "".join(valid_lines)
    df = pd.read_csv(StringIO(data_str), sep=sep, decimal=decimal, engine='python')
    return df

try:
    df = robust_csv_reader(uploaded_file)
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("Spalten in der Datei:", df.columns.tolist())

zielspalten = st.multiselect("🎯 Zielgrößen auswählen (numerische Spalten)", options=df.columns.tolist())
if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

moegliche_features = [c for c in df.columns if c not in zielspalten]
feature_spalten = st.multiselect("🔧 Einflussgrößen (Features) auswählen", options=moegliche_features, default=moegliche_features)
if not feature_spalten:
    st.warning("Bitte mindestens eine Einflussgröße auswählen.")
    st.stop()

X = df[feature_spalten].copy()
y = df[zielspalten].copy()

kategorisch = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerisch = X.select_dtypes(include=[np.number]).columns.tolist()

st.write(f"📊 Kategorische Features: {kategorisch}")
st.write(f"🔢 Numerische Features: {numerisch}")

X_encoded = pd.get_dummies(X)

df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Trainingsdaten nach Bereinigung.")
    st.stop()

modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)
st.success("✅ Modell trainiert!")

st.sidebar.header("🔧 Eingabewerte anpassen")

user_input = {}
for col in numerisch:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))
