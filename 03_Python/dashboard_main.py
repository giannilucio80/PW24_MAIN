"""PW24 MAIN S.p.A. - dashboard didattica basata su dati simulati."""
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PW24 - MAIN S.p.A.", page_icon="📊", layout="wide")

PERCORSO_CSV = Path(__file__).resolve().parent.parent / "02_Dati" / "domanda_mensile_simulata.csv"
PRODOTTI = ["Foodpack", "Supersoftcare", "Softcare Fast", "Medicare"]
PESI = (0.20, 0.30, 0.50)
MESI_VERIFICA = 6
GIORNI_MESE = 30  # convenzione didattica
Z_95 = 1.645  # quantile normale unilaterale per livello di servizio 95%


def carica_dati(percorso):
    if not percorso.is_file():
        raise FileNotFoundError(f"File CSV non trovato: {percorso}")
    dati = pd.read_csv(percorso, sep=";", encoding="utf-8-sig")
    colonne = {"Anno", "Mese", "Prodotto", "Domanda_kg"}
    mancanti = colonne - set(dati.columns)
    if mancanti:
        raise ValueError("Colonne mancanti nel CSV: " + ", ".join(sorted(mancanti)))
    for col in ("Anno", "Mese", "Domanda_kg"):
        dati[col] = pd.to_numeric(dati[col], errors="raise")
    if dati[list(colonne)].isna().any().any():
        raise ValueError("Il CSV contiene valori mancanti nei campi obbligatori.")
    if not dati["Mese"].between(1, 12).all():
        raise ValueError("I mesi devono essere compresi tra 1 e 12.")
    if (dati["Domanda_kg"] < 0).any():
        raise ValueError("La domanda non può essere negativa.")
    dati["Data"] = pd.to_datetime(dict(year=dati["Anno"], month=dati["Mese"], day=1))
    if dati.duplicated(subset=["Prodotto", "Data"]).any():
        raise ValueError("Il CSV contiene mesi duplicati per lo stesso prodotto.")
    return dati.sort_values(["Prodotto", "Data"])


def prevedi(serie):
    """Tre previsioni a un passo, usando solo lo storico passato."""
    y = np.asarray(serie, dtype=float)
    if len(y) < 3:
        raise ValueError("Sono necessari almeno tre mesi di storico.")
    x = np.arange(1, len(y) + 1, dtype=float)
    coeff, intercetta = np.polyfit(x, y, 1)
    return {
        "Media mobile semplice": float(np.mean(y[-3:])),
        "Media mobile ponderata": float(np.dot(y[-3:], PESI)),
        "Regressione lineare": float(coeff * (len(y) + 1) + intercetta),
    }


def confronta_mae(serie, mesi=MESI_VERIFICA):
    """Backtest expanding-window: ogni mese usa solo dati precedenti."""
    y = np.asarray(serie, dtype=float)
    if len(y) < mesi + 3:
        raise ValueError(f"Servono almeno {mesi + 3} osservazioni per calcolare il MAE.")
    errori = {metodo: [] for metodo in prevedi(y[:-mesi])}
    for i in range(len(y) - mesi, len(y)):
        previsioni = prevedi(y[:i])
        for metodo, valore in previsioni.items():
            errori[metodo].append(abs(float(y[i]) - valore))
    return {metodo: float(np.mean(valori)) for metodo, valori in errori.items()}


st.title("PW 24 - MAIN S.p.A.")
st.subheader("Previsione della domanda e pianificazione delle scorte")
st.info("Prototipo universitario: domanda, giacenze e parametri numerici sono simulati, non dati gestionali MAIN.")

try:
    df = carica_dati(PERCORSO_CSV)
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Impossibile leggere il dataset: {exc}")
    st.stop()

st.sidebar.header("Parametri di analisi")
prodotto = st.sidebar.selectbox("Seleziona il prodotto", PRODOTTI)
dati_prodotto = df[df["Prodotto"] == prodotto].sort_values("Data").copy()
if len(dati_prodotto) < MESI_VERIFICA + 3:
    st.error(f"Per {prodotto} servono almeno {MESI_VERIFICA + 3} mesi di dati; trovati {len(dati_prodotto)}.")
    st.stop()

serie = dati_prodotto["Domanda_kg"].to_numpy(dtype=float)
try:
    previsioni_prodotto = prevedi(serie)
    mae_prodotto = confronta_mae(serie)
except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
    st.error(f"Impossibile calcolare le previsioni: {exc}")
    st.stop()

metodo_migliore = min(mae_prodotto, key=mae_prodotto.get)
previsione_scelta = previsioni_prodotto[metodo_migliore]
periodo_previsto = (dati_prodotto["Data"].max() + pd.offsets.MonthBegin(1)).strftime("%m/%Y")
inizio_verifica = dati_prodotto["Data"].iloc[-MESI_VERIFICA].strftime("%m/%Y")
fine_verifica = dati_prodotto["Data"].iloc[-1].strftime("%m/%Y")

st.header(f"Analisi del prodotto: {prodotto}")
c1, c2 = st.columns(2)
c1.metric("Domanda totale del periodo", f"{serie.sum():,.0f} kg")
c2.metric("Domanda media mensile", f"{serie.mean():,.0f} kg")
st.subheader("Andamento storico della domanda")
st.line_chart(dati_prodotto.set_index("Data")["Domanda_kg"])
with st.expander("Visualizza i dati mensili"):
    st.dataframe(dati_prodotto[["Anno", "Mese", "Prodotto", "Domanda_kg"]], hide_index=True, use_container_width=True)

st.divider()
st.header(f"Previsione della domanda - {periodo_previsto}")
tabella_previsioni = pd.DataFrame(list(previsioni_prodotto.items()), columns=["Metodo previsionale", "Previsione (kg)"])
st.dataframe(tabella_previsioni, hide_index=True, use_container_width=True,
             column_config={"Previsione (kg)": st.column_config.NumberColumn("Previsione (kg)", format="%.0f")})
st.subheader("Confronto delle previsioni")
st.bar_chart(tabella_previsioni.set_index("Metodo previsionale"))
st.caption("Tutte le previsioni sono calcolate direttamente dal CSV con dati simulati.")

st.divider()
st.header("Confronto dell'accuratezza dei metodi")
st.write(f"MAE: media degli errori assoluti delle previsioni a un mese, sul periodo {inizio_verifica}–{fine_verifica}. Più basso è il MAE, minore è l'errore storico.")
tabella_mae = pd.DataFrame(list(mae_prodotto.items()), columns=["Metodo previsionale", "MAE (kg)"])
st.dataframe(tabella_mae, hide_index=True, use_container_width=True,
             column_config={"MAE (kg)": st.column_config.NumberColumn("MAE (kg)", format="%.0f")})
st.bar_chart(tabella_mae.set_index("Metodo previsionale"))
st.success(f"Metodo con MAE minore per {prodotto}: {metodo_migliore} ({mae_prodotto[metodo_migliore]:,.0f} kg)")
st.metric(f"Previsione {periodo_previsto} del metodo selezionato", f"{previsione_scelta:,.0f} kg")
st.caption("Il metodo con MAE minore negli ultimi sei mesi non è necessariamente il migliore in futuro.")

st.divider()
st.header("Simulazione della pianificazione delle scorte")
st.write("Il modello usa la previsione del metodo con MAE minore. Le variabili modificabili descrivono scenari ipotetici.")
lead_time = st.slider("Lead time di reintegro (giorni)", 1, 30, 10)
variabilita = st.slider("Variabilità giornaliera della domanda (%)", 5, 50, 20)
lotto_produzione = st.number_input("Lotto di produzione (kg)", min_value=100, max_value=100000, value=6000, step=500)
produzione_anticipata = st.number_input("Produzione anticipata (kg)", min_value=0, max_value=100000, value=3000, step=500)
trend = st.slider("Variazione della domanda prevista (%)", -30, 30, 0, 5)
stagionalita = st.slider("Effetto stagionale sulla domanda (%)", -30, 30, 0, 5)

domanda_simulata = previsione_scelta * (1 + trend / 100) * (1 + stagionalita / 100)
domanda_giornaliera = domanda_simulata / GIORNI_MESE
deviazione_giornaliera = domanda_giornaliera * variabilita / 100
scorta_ciclo = lotto_produzione / 2
scorta_sicurezza = Z_95 * deviazione_giornaliera * sqrt(lead_time)
scorta_anticipata = produzione_anticipata
fabbisogno_lead_time = domanda_giornaliera * lead_time
punto_riordino = fabbisogno_lead_time + scorta_sicurezza

st.metric("Domanda mensile simulata", f"{domanda_simulata:,.0f} kg")
st.subheader("Risultati della simulazione")
c1, c2 = st.columns(2)
c1.metric("Scorta di ciclo media", f"{scorta_ciclo:,.0f} kg")
c1.metric("Scorta di sicurezza", f"{scorta_sicurezza:,.0f} kg")
c2.metric("Scorta anticipata", f"{scorta_anticipata:,.0f} kg")
c2.metric("Pipeline (proxy: fabbisogno nel lead time)", f"{fabbisogno_lead_time:,.0f} kg")
st.metric("Punto di riordino teorico", f"{punto_riordino:,.0f} kg")
st.subheader("Confronto delle tipologie di scorte")
tabella_scorte = pd.DataFrame({"Tipologia": ["Ciclo", "Sicurezza", "Anticipata", "Pipeline (proxy)"],
                              "Quantità (kg)": [scorta_ciclo, scorta_sicurezza, scorta_anticipata, fabbisogno_lead_time]})
st.bar_chart(tabella_scorte.set_index("Tipologia"))
st.caption("Modello semplificato: scorta di ciclo media = lotto/2; sicurezza = z × deviazione giornaliera × √lead time, con domanda indipendente, lead time fisso e livello di servizio teorico del 95%. La voce pipeline usa il fabbisogno durante il lead time come proxy e NON misura materiale realmente in transito o in lavorazione. La scorta anticipata può essere già compresa nella giacenza fisica. Le quattro voci non sono sommabili automaticamente.")

st.divider()
st.header("Confronto tra scenario iniziale e simulazione")
confronto = pd.DataFrame({"Scenario": ["Previsione iniziale", "Domanda simulata"],
                         "Domanda (kg)": [previsione_scelta, domanda_simulata]})
st.bar_chart(confronto.set_index("Scenario"))
st.metric("Variazione complessiva della domanda", f"{(domanda_simulata / previsione_scelta - 1) * 100:+.1f}%" if previsione_scelta > 0 else "Non calcolabile (previsione zero)")
st.caption("Trend e stagionalità sono fattori percentuali di scenario applicati in modo moltiplicativo, non parametri stimati statisticamente in questa sezione.")

st.caption("La simulazione riguarda quantità aggregate per prodotto e non gestisce giacenze attribuite ai singoli clienti. Il punto di riordino è soltanto un riferimento teorico, non una decisione automatica di produzione.")
