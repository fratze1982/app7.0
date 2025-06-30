import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flexible KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 Flexible KI-Vorhersage für Lackrezepturen")

uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

def robust_csv_reader(file, sep=';', decimal=','):
    bad_lines = []
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

zielspalten = st.multiselect("🎯 Zielgrößen auswählen (numerische Spalten)", options=df.select_dtypes(include=[np.number]).columns.tolist())
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

# --- Zieloptimierung ---
ergebnis_df = pd.DataFrame()

st.subheader("🎯 Zieloptimierung")
if zielspalten:
    zielwerte = {}
    toleranzen = {}
    gewichtung = {}

    for ziel in zielspalten:
        zielwerte[ziel] = st.number_input(f"Zielwert für {ziel}", value=float(df[ziel].mean()))
        toleranzen[ziel] = st.number_input(f"Toleranz für {ziel} (±)", value=2.0 if "Glanz" in ziel else 1.0)
        gewichtung[ziel] = st.slider(f"Gewichtung für {ziel}", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    anzahl_varianten = 500
    simulierte_formulierungen = []
    score_liste = []

    if st.button("🚀 Starte Zielsuche"):
        for _ in range(anzahl_varianten):
            zufall = {}
            for roh in numerisch:
                min_val = float(df[roh].min())
                max_val = float(df[roh].max())
                zufall[roh] = np.random.uniform(min_val, max_val)
            simulierte_formulierungen.append(zufall)

        sim_df = pd.DataFrame(simulierte_formulierungen)
        sim_encoded = pd.get_dummies(sim_df)
        for col in X_clean.columns:
            if col not in sim_encoded.columns:
                sim_encoded[col] = 0
        sim_encoded = sim_encoded[X_clean.columns]

        y_pred = modell.predict(sim_encoded)

        treffer_idx = []
        for i, y in enumerate(y_pred):
            score = 0
            passt = True
            for ziel in zielspalten:
                delta = abs(y[zielspalten.index(ziel)] - zielwerte[ziel])
                score += delta * gewichtung[ziel]
                if delta > toleranzen[ziel]:
                    passt = False
            if passt:
                score_liste.append((i, score))

        if score_liste:
            score_liste.sort(key=lambda x: x[1])
            treffer_idx = [i for i, s in score_liste]

            treffer_df = sim_df.iloc[treffer_idx].copy()
            vorhersagen_df = pd.DataFrame(
                [y_pred[i] for i in treffer_idx],
                columns=zielspalten
            )

            ergebnis_df = pd.concat(
                [treffer_df.reset_index(drop=True), vorhersagen_df.reset_index(drop=True)],
                axis=1
            )
            ergebnis_df.insert(0, "Score", [round(s, 2) for _, s in score_liste])

            st.success(f"✅ {len(ergebnis_df)} passende Formulierungen gefunden!")
            st.dataframe(ergebnis_df)

# --- Vergleich und Diagramm ---
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
