import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# CSV-Daten laden
df = pd.read_csv("rezeptdaten.csv", encoding="latin1")

# Zielgrößen definieren
targets = [
    "Glanz 20", "Glanz 60", "Glanz 85",
    "Viskosität lowshear", "Viskosität midshear", "Brookfield",
    "Kosten Gesamt kg"
]

# Eingabe- und Ausgabedaten trennen
X = df.drop(columns=targets)
y = df[targets]

# Kategorische Variablen erkennen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# Modell trainieren
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_encoded, y)

# Streamlit-UI
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# Eingabeformular
user_input = {}
st.sidebar.header("🔧 Eingabewerte anpassen")
for col in numerisch:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

# Eingabe vorbereiten
input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten ergänzen
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# Vorhersage
prediction = modell.predict(input_encoded)[0]

# Ergebnisse anzeigen
st.subheader("🔮 Vorhergesagte Eigenschaften")
for i, ziel in enumerate(targets):
    st.metric(label=ziel, value=round(prediction[i], 2))
