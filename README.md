# PW24 - Previsione della domanda e pianificazione delle scorte

## Descrizione

Prototipo universitario sviluppato in Python con Streamlit
per la previsione della domanda e la simulazione delle scorte
in un contesto di produzione industriale.

Il progetto prende come riferimento il settore produttivo
di MAIN S.p.A., azienda specializzata nella produzione
di materiali airlaid multistrato.

Tutti i dati relativi alla domanda e alle scorte sono simulati
e non rappresentano informazioni gestionali reali dell'azienda.

## Prodotti analizzati

- Foodpack
- Supersoftcare
- Softcare Fast
- Medicare

## Funzionalità

La dashboard consente di:

- Visualizzare l'andamento storico della domanda.
- Calcolare quattro metodi previsionali.
- Confrontare l'accuratezza mediante il MAE.
- Selezionare il metodo con MAE minore.
- Simulare quattro tipologie di scorte.
- Modificare lead time e variabilità della domanda.
- Simulare variazioni tendenziali e stagionali.
- Distinguere giacenza disponibile e materiale riservato.
- Confrontare la disponibilità con una soglia teorica di riordino.

## Metodi previsionali

1. Media mobile semplice a tre mesi.
2. Media mobile ponderata a tre mesi.
3. Smorzamento esponenziale semplice.
4. Regressione lineare.

Il MAE viene calcolato sugli ultimi sei mesi
del dataset mediante previsioni a un passo.

## Tecnologie

- Python
- Streamlit
- Pandas
- NumPy

## Struttura delle cartelle

PW24_MAIN/
    02_Dati/
        domanda_mensile_simulata.csv
    03_Python/
        dashboard_main.py
        requirements.txt

## Installazione

È necessario avere Python installato.

Aprire il Prompt dei comandi nella cartella 03_Python
ed eseguire:

    python -m pip install -r requirements.txt

## Avvio

Dalla cartella 03_Python eseguire:

    python -m streamlit run dashboard_main.py

La dashboard viene visualizzata nel browser,
generalmente all'indirizzo http://localhost:8501

## Note metodologiche

I dati sono interamente simulati.

Le previsioni non costituiscono stime della domanda reale
di MAIN S.p.A.

Le formule relative alle scorte rappresentano
un modello didattico semplificato.

Le variazioni di trend e stagionalità sono applicate
come scenari percentuali alla previsione iniziale.

La soglia di riordino è un indicatore teorico
e non rappresenta una procedura aziendale effettiva.

## Finalità

Project Work universitario - Corso di Laurea L-31
Informatica per le Aziende Digitali.