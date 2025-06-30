import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("\U0001F3A8 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("\U0001F4C1 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen mit Fehlerbehandlung ---
try:
    content = uploaded_file.getvalue().decode("utf-8").splitlines()
    header_count = content[0].count(";")
    clean_lines = [line for line in content if line.count(";") == header_count]
    df = pd.read_csv(StringIO("\n".join(clean_lines)), sep=";", decimal=",")
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("\U0001F9FE Gefundene Spalten:", df.columns.tolist())

# --- Zielgrößen ---
numerische_spalten = df.select_dtypes(include=[np.number]).columns.tolist()
if not numerische_spalten:
    st.error("❌ Keine numerischen Spalten im Datensatz gefunden.")
    st.stop()

zielspalten = st.multiselect("\U0001F3AF Zielgrößen auswählen (numerisch)", options=numerische_spalten, default=[numerische_spalten[0]])
if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

X = df.drop(columns=zielspalten)
y = df[zielspalten]

kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

X_encoded = pd.get_dummies(X)
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()
X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

st.sidebar.header("\U0001F527 Parameter anpassen")
user_input = {}
for col in numerisch:
    user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
for col in kategorisch:
    user_input[col] = st.sidebar.selectbox(col, sorted(df[col].dropna().unique()))

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

prediction = modell.predict(input_encoded)[0]
st.subheader("\U0001F52E Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(ziel, round(prediction[i], 2))

# --- Zieloptimierung ---
st.subheader("\U0001F3AF Zieloptimierung per Zufallssuche")
zielwerte = {}
toleranzen = {}
gewichtung = {}

with st.expander("⚙️ Zielwerte und Toleranzen setzen"):
    for ziel in zielspalten:
        zielwerte[ziel] = st.number_input(f"Zielwert für {ziel}", value=float(df[ziel].mean()))
        toleranzen[ziel] = st.number_input(f"Toleranz (±) für {ziel}", value=2.0)
        gewichtung[ziel] = st.slider(f"Gewichtung für {ziel}", 0.0, 5.0, 1.0, 0.1)

steuerbare_rohstoffe = [col for col in numerisch if col in df.columns]
st.sidebar.header("\U0001F6E0️ Rohstoffe fixieren oder begrenzen")
fixierte_werte = {}
rohstoffgrenzen = {}
for roh in steuerbare_rohstoffe:
    if st.sidebar.checkbox(f"{roh} fixieren?"):
        fixierte_werte[roh] = st.sidebar.number_input(f"Wert für {roh}", value=float(df[roh].mean()))
    else:
        min_val, max_val = float(df[roh].min()), float(df[roh].max())
        if min_val == max_val:
            min_val -= 0.01
            max_val += 0.01
        rohstoffgrenzen[roh] = st.sidebar.slider(f"Grenzen für {roh}", min_val, max_val, (min_val, max_val))

if st.button("\U0001F680 Zielsuche starten"):
    varianten = 1000
    formulierungsliste = []
    score_liste = []
    for _ in range(varianten):
        probe = {}
        for roh in steuerbare_rohstoffe:
            if roh in fixierte_werte:
                probe[roh] = fixierte_werte[roh]
            else:
                min_r, max_r = rohstoffgrenzen.get(roh, (df[roh].min(), df[roh].max()))
                probe[roh] = np.random.uniform(min_r, max_r)
        formulierungsliste.append(probe)

    sim_df = pd.DataFrame(formulierungsliste)
    sim_encoded = pd.get_dummies(sim_df)
    for col in X_clean.columns:
        if col not in sim_encoded.columns:
            sim_encoded[col] = 0
    sim_encoded = sim_encoded[X_clean.columns]
    y_pred = modell.predict(sim_encoded)

    ergebnisse = []
    for i, row in enumerate(y_pred):
        score = 0
        passt = True
        for ziel in zielspalten:
            delta = abs(row[zielspalten.index(ziel)] - zielwerte[ziel])
            score += delta * gewichtung[ziel]
            if delta > toleranzen[ziel]:
                passt = False
        if passt:
            ergebnisse.append((i, score))

    if ergebnisse:
        ergebnisse.sort(key=lambda x: x[1])
        idxs = [i for i, _ in ergebnisse]
        ergebnis_df = sim_df.iloc[idxs].reset_index(drop=True)
        ergebnis_df["Score"] = [round(s, 2) for _, s in ergebnisse]
        for j, ziel in enumerate(zielspalten):
            ergebnis_df[ziel] = [y_pred[i][j] for i in idxs]
        st.success(f"✅ {len(ergebnis_df)} passende Formulierungen gefunden")
        st.dataframe(ergebnis_df)

        csv = ergebnis_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV herunterladen", data=csv, file_name="formulierungsergebnisse.csv")

        if len(ergebnis_df) >= 3:
            st.subheader("🔬 Radar-Diagramm der Top 3")
            top3 = ergebnis_df.head(3)
            labels = zielspalten + [zielspalten[0]]
            angles = np.linspace(0, 2*np.pi, len(zielspalten), endpoint=False).tolist()
            angles += angles[:1]
            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
            for idx, row in top3.iterrows():
                values = row[zielspalten].tolist() + [row[zielspalten[0]]]
                ax.plot(angles, values, label=f"Formulierung {idx+1}")
                ax.fill(angles, values, alpha=0.1)
            ax.set_thetagrids(np.degrees(angles), labels)
            ax.set_title("Radarvergleich Zielgrößen")
            ax.legend()
            st.pyplot(fig)

        st.subheader("📊 Balkendiagramm-Vergleich")
        max_auswahl = min(len(ergebnis_df), 10)
        index_auswahl = st.multiselect("Wähle Formulierungen zum Vergleich", list(range(len(ergebnis_df))), default=list(range(min(3, len(ergebnis_df)))))
        if index_auswahl:
            vergleich_df = ergebnis_df.loc[index_auswahl, zielspalten].copy()
            vergleich_df["Formulierung"] = [f"F{i+1}" for i in index_auswahl]
            vergleich_df = vergleich_df.set_index("Formulierung")
            fig2, ax2 = plt.subplots(figsize=(10,6))
            vergleich_df.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Wert")
            ax2.set_title("Zielgrößen-Vergleich ausgewählter Formulierungen")
            ax2.legend(title="Zielgröße")
            st.pyplot(fig2)
    else:
        st.warning("❌ Keine passenden Formulierungen innerhalb der Toleranzen gefunden.")
