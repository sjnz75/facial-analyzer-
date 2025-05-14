# face-aesthetics-analyzer

Analisi estetica facciale: selezione landmarks, calcoli clinici e diagnostica.

## Installazione
```bash
git clone https://github.com/tuo-username/face-aesthetics-analyzer.git
cd face-aesthetics-analyzer
pip install -r requirements.txt
streamlit run app.py
uvicorn api.main:app --reload --port 8000
