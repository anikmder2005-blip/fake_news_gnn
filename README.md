# Fake News Detector (graph neural network), upgraded app

A graph convolutional network compares a news article with 6,335 labelled
articles and predicts whether it looks real or fake.

## Inputs

| Tab | What you can add | Needs |
|---|---|---|
| Text | Pasted text, or a built-in real or fake sample | core |
| File | .txt, .md, .pdf (selectable text), .docx, .html | core |
| Image | Screenshot or photo of an article (text is read with OCR) | optional: rapidocr-onnxruntime |
| Video and audio | Video (on-screen text from frames), audio or video speech | optional: rapidocr-onnxruntime, faster-whisper |
| Link | A news article web address | core, internet |
| Batch (CSV) | One article per row, results downloadable as CSV | core |

## Set up (once)

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt --no-cache-dir
python -m pip install -r requirements-optional.txt
```

The last line is only for image, video and speech. Skip it if you only need
text, files, links and CSV. If PowerShell blocks `activate`, run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.
If PyTorch will not install, use Python 3.12 (very new Python versions can lack PyTorch builds).

## Run

```
python -m streamlit run app/streamlit_app.py
```

or double-click `run_app.bat`. The app opens at http://localhost:8501.

## Change the background

Open the sidebar. Pick Network night, Deep teal or Violet dusk, choose
"My own image" to upload your own picture, or pick None. The "Panel opacity"
slider controls how much of the background shows through.

## Read the result carefully

* The model was trained on US political news from one period. On other topics it
  can answer "fake" with very high confidence. The app warns when an article is
  unlike the training data (similarity below 0.30) or too short (under 30 words).
* It judges writing style and similarity to known articles. It does not check facts.
* The built-in samples come from the training data, so they look easier than a new article.

## Test results (1,267 unseen articles)

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 90.6% | 90.7% |
| Random Forest | 91.2% | 91.1% |
| GCN (this app) | 88.5% | 88.6% |

## Folders

```
app/            streamlit_app.py (the interface)
src/            preprocess, build_graph, model, train, predictor, extractors, backgrounds
data/           graph files, vectorizer, article titles, raw dataset
models/         gcn_model.pt (trained weights)
results/        metrics and charts
assets/         background images
samples/        demo articles
.streamlit/     dark theme and upload size
```

To retrain: `cd src`, then run `python preprocess.py`, `python build_graph.py`, `python train.py`.
