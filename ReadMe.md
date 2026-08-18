# Setup

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. Run the app

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## First-run notes

- **Pretrained models download automatically** the first time each feature is
  used (Meeting Summary, short labels, chat Q&A, audio transcription).
  Roughly 4-5GB total, needs internet, one-time only — they're cached at
  `~/.cache/huggingface/` and reused after that.
- **The decision classifier** (`models/decision_classifier/`) is already
  trained and included in this folder — no need to retrain it.
  If it's missing, run `python3 train_classifier.py` first (needs the AMI
  dataset under `data/raw/`, downloaded via `python3 download_datasets.py`).

## NOTE 
 
 To Run and Execute the Source code and project you need to download the Data and Models provided in the below links
 
 Data : https://drive.google.com/drive/folders/1Wg4qpVqPFBL0oVsZQbq9bRBdcJnsPpeZ?usp=drive_link
 Models : https://drive.google.com/drive/folders/1vFCE4KNGlDsg9plw6KeQdB8ld8I2jbpU?usp=drive_link
