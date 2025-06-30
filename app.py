import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("\U0001F3A8 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("\U0001F4C1 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen mit Fehlerbehandlung für uneinheitliche Zeilen ---
try:
    df_raw = uploaded_file.getvalue().decode("utf-8").splitlines()
    header = df_raw[0].count(";")
    df_cleaned = [line for line in df_raw if line.count(";") == header]
    df = pd.read_csv(pd.compat.StringIO("\n".join(df_cleaned)), sep=";", decimal=",")
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("\U0001F9FE Gefundene Spalten:", df.columns.tolist())

# --- Zielgrößen aus numerischen Spalten dynamisch auswählen ---
numerische_spalten = df.select_dtypes(include=[np.number]).columns.tolist()

if not numerische_spalten:
    st.error("❌ Keine numerischen Spalten im Datensatz gefunden.")
    st.stop()

zielspalten = st.multiselect(
    "\U0001F3AF Zielgrößen auswählen (numerische Spalten)",
    options=numerische_spalten,
    default=[numerische_spalten[0]]
)

if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

# --- Eingabe- und Zielvariablen trennen ---
X = df.drop(columns=zielspalten, errors="ignore")
y = df[zielspalten].copy()

# Spaltentypen bestimmen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# Fehlende Werte bereinigen
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Daten zum Trainieren.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("\U0001F527 Parameter anpassen")
user_input = {}

for col in numerisch:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    except:
        continue

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

st.subheader("\U0001F52E Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Interaktives Balkendiagramm nur anzeigen, wenn ergebnis_df existiert ---
if 'ergebnis_df' in locals() and not ergebnis_df.empty:
    st.subheader("📊 Vergleich ausgewählter Formulierungen als Balkendiagramm")
    max_auswahl = min(len(ergebnis_df), 10)
    index_auswahl = st.multiselect(
        "Wähle bis zu 5 Formulierungen zum Vergleich:",
        options=list(range(len(ergebnis_df))),
        default=list(range(min(3, len(ergebnis_df)))),
        help="Die Auswahl basiert auf den Zeilenpositionen aus der Tabelle."
    )

    if index_auswahl:
        vergleich_df = ergebnis_df.loc[index_auswahl, zielspalten].copy()
        vergleich_df["Formulierung"] = [f"F{i+1}" for i in index_auswahl]
        vergleich_df = vergleich_df.set_index("Formulierung")

        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        vergleich_df.plot(kind="bar", ax=ax_bar)
        ax_bar.set_ylabel("Wert")
        ax_bar.set_title("Zielgrößen-Vergleich ausgewählter Formulierungen")
        ax_bar.legend(title="Zielgröße")
        st.pyplot(fig_bar)
else:
    st.info("ℹ️ Noch keine Ergebnisse vorhanden. Bitte zuerst die Zielsuche starten.")
