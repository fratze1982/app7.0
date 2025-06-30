import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(page_title="Flexible KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 Flexible KI-Vorhersage für Lackrezepturen")

# --- Datei Upload ---
uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV dynamisch einlesen (automatisches Trennzeichen) ---
try:
    df = pd.read_csv(uploaded_file, sep=None, engine="python", decimal=",")
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("Spalten in der Datei:", df.columns.tolist())

# --- Zielgrößen auswählen ---
zielspalten = st.multiselect("🎯 Zielgrößen auswählen (numerische Spalten)", options=df.columns.tolist())
if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

# --- Features auswählen (alle anderen Spalten außer Zielspalten) ---
moegliche_features = [c for c in df.columns if c not in zielspalten]
feature_spalten = st.multiselect("🔧 Einflussgrößen (Features) auswählen", options=moegliche_features, default=moegliche_features)

if not feature_spalten:
    st.warning("Bitte mindestens eine Einflussgröße auswählen.")
    st.stop()

# --- Features und Zielwerte ---
X = df[feature_spalten].copy()
y = df[zielspalten].copy()

# --- Kategorische und numerische Features erkennen ---
kategorisch = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerisch = X.select_dtypes(include=[np.number]).columns.tolist()

st.write(f"📊 Kategorische Features: {kategorisch}")
st.write(f"🔢 Numerische Features: {numerisch}")

# --- One-Hot-Encoding für kategorische Features ---
X_encoded = pd.get_dummies(X)

# --- Fehlende Werte entfernen ---
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Trainingsdaten nach Bereinigung.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)
st.success("✅ Modell trainiert!")

# --- Eingabeformular für Vorhersage ---
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

# Fehlende Spalten auffüllen
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))
