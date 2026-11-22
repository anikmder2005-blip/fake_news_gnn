"""
Fake News Detector (graph neural network)

Inputs: pasted text, files (txt, pdf, docx, html), images, video, audio,
web links, and CSV batches. Pick a background in the sidebar or upload your own.

Run from the project folder:
    python -m streamlit run app/streamlit_app.py
"""
import io
import json
import os
import random
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import backgrounds as bg
import extractors as ex

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

LOW_SIMILARITY = 0.30   # below this, the article looks unlike the training data
MIN_WORDS = 30          # below this, the text is too short to judge
RED, GREEN, BLUE = "#E5484D", "#30A46C", "#5B9DFF"


# ----------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the model (first time only)")
def get_predictor():
    from predictor import Predictor
    return Predictor()


@st.cache_data
def preset_uri(name: str) -> str:
    return bg.to_css_data_uri(bg.preset_bytes(name))


@st.cache_data
def load_samples() -> dict:
    path = os.path.join(BASE, "samples", "sample_articles.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {"FAKE": [], "REAL": []}


@st.cache_data
def load_metrics() -> dict:
    try:
        with open(os.path.join(BASE, "results", "metrics.json")) as f:
            return json.load(f)
    except OSError:
        return {}


st.session_state.setdefault("text_input", "")
st.session_state.setdefault("sample_text", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("batch", None)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Appearance")
    choice = st.selectbox("Background", list(bg.PRESETS) + ["My own image", "None"])
    custom = None
    if choice == "My own image":
        custom = st.file_uploader("Upload a background", type=["png", "jpg", "jpeg", "webp"],
                                  key="bg_upload")
    opacity = st.slider("Panel opacity", 0.30, 0.95, 0.78, 0.01,
                        help="Lower values show more of the background behind the content.")

    st.header("Model")
    gcn = load_metrics().get("GCN (Ours)")
    if gcn:
        st.caption(f"Graph convolutional network: {gcn['accuracy']*100:.1f}% accuracy and "
                   f"{gcn['f1']*100:.1f}% F1 on 1,267 unseen articles.")
    st.caption("Trained on US political news from one period. Other topics are less reliable.")

    st.header("Optional features")
    for name, ok in ex.optional_status().items():
        st.write(("✅ " if ok else "⬜ ") + name)
    st.caption("To switch one on, install it with pip. The tab for that input shows the exact command.")

    if st.session_state.history:
        st.header("Recent checks")
        for h in st.session_state.history[:8]:
            st.caption(f"{h['label'].title()} at {h['confidence']*100:.0f}% · {h['source']}")

if choice == "None":
    uri = None
elif choice == "My own image":
    uri = bg.to_css_data_uri(custom.getvalue()) if custom else preset_uri("Network night")
else:
    uri = preset_uri(choice)
st.markdown(bg.page_css(uri, opacity), unsafe_allow_html=True)


# ----------------------------------------------------------------- actions
def run_analysis(text: str, source: str):
    predictor = get_predictor()
    with st.spinner("Checking the article"):
        pred = predictor.predict(text)
    is_sample = bool(st.session_state.sample_text) and \
        text.strip() == st.session_state.sample_text.strip()
    st.session_state.result = {
        "source": source, "text": text, "pred": pred,
        "is_sample": is_sample, "time": time.strftime("%H:%M:%S"),
    }
    st.session_state.history.insert(0, {
        "source": source, "label": pred["label"], "confidence": pred["confidence"]})
    st.session_state.history = st.session_state.history[:20]


def guarded(fn):
    """Run fn and show friendly messages for the usual failure types."""
    try:
        return fn()
    except ex.MissingDependency as e:
        st.warning(f"{e.feature} needs one extra library. Run this in the terminal, "
                   f"then restart the app:")
        st.code(e.pip_cmd, language="bash")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:  # network, corrupt files and similar
        st.error(f"Something went wrong: {e}")
    return None


def load_sample(label: str):
    items = load_samples().get(label, [])
    if items:
        item = random.choice(items)
        st.session_state.text_input = item["text"]
        st.session_state.sample_text = item["text"]


# ----------------------------------------------------------------- results
def draw_neighbours(neighbours: list, verdict: str):
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    n = len(neighbours)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    centre_col = RED if verdict == "FAKE" else GREEN
    for ang, nb in zip(angles, neighbours):
        x, y = np.cos(ang), np.sin(ang)
        col = RED if nb["label"] == "FAKE" else GREEN
        ax.plot([0, x], [0, y], color="#8FA3C7", alpha=0.35 + 0.6 * nb["similarity"],
                lw=1 + 5 * nb["similarity"], zorder=1)
        ax.scatter([x], [y], s=520, color=col, zorder=2, edgecolor="white", linewidth=0.8)
        title = nb["title"] if len(nb["title"]) <= 40 else nb["title"][:39] + "…"
        ax.text(x * 1.52, y * 1.40, textwrap.fill(title, 22) + f"\n{nb['label'].title()}, "
                f"similarity {nb['similarity']:.2f}", ha="center", va="center",
                fontsize=10, color="#E6EAF2")
    ax.scatter([0], [0], s=1500, color=BLUE, zorder=3, edgecolor=centre_col, linewidth=3)
    ax.text(0, 0, "Your\narticle", ha="center", va="center", fontsize=10,
            color="white", fontweight="bold", zorder=4)
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(-1.95, 1.95)
    ax.axis("off")
    fig.tight_layout()
    return fig


def show_result(res: dict):
    pred = res["pred"]
    fake = pred["label"] == "FAKE"
    st.subheader("Result")
    st.markdown(
        f"""<div class="verdict {'fake' if fake else 'real'}">
        <div class="small">Model prediction · {res['source']}</div>
        <div class="big">{'Potentially fake' if fake else 'Likely real'}</div>
        <div class="conf">Confidence {pred['confidence']*100:.1f}%</div></div>""",
        unsafe_allow_html=True)

    if pred["max_similarity"] < LOW_SIMILARITY:
        st.warning("This text is quite different from the 6,335 articles the model learned from "
                   f"(closest match {pred['max_similarity']:.2f}). Treat the result as unreliable, "
                   "even when the confidence is high.")
    if pred["n_words"] < MIN_WORDS:
        st.warning(f"Only {pred['n_words']} useful words were found. Longer text gives a steadier result.")
    if res.get("is_sample"):
        st.info("This sample comes from the training data, so the model has seen it before. "
                "Use a new article to test it properly.")

    c1, c2 = st.columns(2)
    c1.metric("Probability fake", f"{pred['p_fake']*100:.1f}%")
    c2.metric("Probability real", f"{pred['p_real']*100:.1f}%")

    st.markdown("**Closest known articles**")
    buf = io.BytesIO()
    fig = draw_neighbours(pred["neighbours"], pred["label"])
    fig.savefig(buf, format="png", transparent=True, dpi=170, bbox_inches="tight")
    plt.close(fig)
    st.image(buf.getvalue(), width=760)
    st.caption("Red nodes are known fake articles, green nodes are known real ones. "
               "Thicker lines mean more similar text.")
    table = pd.DataFrame([{"Title": n["title"][:90], "Label": n["label"].title(),
                           "Similarity": round(n["similarity"], 2)}
                          for n in pred["neighbours"]])
    st.dataframe(table, hide_index=True)

    with st.expander("Text the model read"):
        st.text(res["text"][:4000] + ("…" if len(res["text"]) > 4000 else ""))

    export = {k: v for k, v in res.items() if k != "text"}
    export["text_preview"] = res["text"][:500]
    st.download_button("Download result (JSON)", json.dumps(export, indent=2),
                       file_name="fake_news_result.json", mime="application/json")
    st.caption("This is a prediction from writing style and similarity to known articles. "
               "It does not check whether the claims are true.")


# ----------------------------------------------------------------- page
st.title("Fake News Detector")
st.write("Add a news article in any form. A graph neural network compares it with 6,335 labelled "
         "articles and predicts whether it looks real or fake.")

tab_text, tab_file, tab_img, tab_vid, tab_url, tab_csv = st.tabs(
    ["Text", "File", "Image", "Video and audio", "Link", "Batch (CSV)"])

# --- text
with tab_text:
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Try a real sample", on_click=load_sample, args=("REAL",))
    b2.button("Try a fake sample", on_click=load_sample, args=("FAKE",))
    st.text_area("Article text", key="text_input", height=220,
                 placeholder="Paste the full article here. A few paragraphs work best.")
    if st.button("Check text", type="primary", key="go_text"):
        if not st.session_state.text_input.strip():
            st.warning("Paste some article text first, or try a sample.")
        else:
            run_analysis(st.session_state.text_input, "Pasted text")

# --- file
with tab_file:
    up = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx", "html"],
                          key="file_up")
    st.caption("Supported: text, Markdown, PDF with selectable text, Word and saved web pages.")
    if st.button("Check file", type="primary", key="go_file", disabled=up is None):
        text = guarded(lambda: ex.extract_from_file(up.name, up.getvalue()))
        if text:
            run_analysis(text, f"File: {up.name}")

# --- image
with tab_img:
    up = st.file_uploader("Upload a screenshot or photo of an article",
                          type=["png", "jpg", "jpeg", "webp", "bmp"], key="img_up")
    if up:
        st.image(up, width=420)
    st.caption("The text in the picture is read automatically. Clear, high-contrast images work best.")
    if st.button("Check image", type="primary", key="go_img", disabled=up is None):
        with st.spinner("Reading the text in the image"):
            text = guarded(lambda: ex.extract_from_image(up.getvalue()))
        if text:
            run_analysis(text, f"Image: {up.name}")

# --- video and audio
with tab_vid:
    up = st.file_uploader("Upload a video or an audio recording",
                          type=[e.strip(".") for e in ex.VIDEO_EXT + ex.AUDIO_EXT], key="vid_up")
    is_audio = bool(up) and os.path.splitext(up.name)[1].lower() in ex.AUDIO_EXT
    if up:
        (st.audio if is_audio else st.video)(up)
    v1, v2 = st.columns(2)
    every = v1.slider("Seconds between frames", 1, 10, 2, disabled=is_audio,
                      help="The app reads on-screen text such as headlines and captions from these frames.")
    max_frames = v2.slider("Most frames to read", 5, 80, 30, disabled=is_audio)
    speech = st.checkbox("Also turn speech into text", value=is_audio, disabled=is_audio,
                         help="Needs faster-whisper. It downloads a small model the first time.")
    st.caption("Videos are read two ways: on-screen text from sampled frames, and optionally speech.")
    if st.button("Check video or audio", type="primary", key="go_vid", disabled=up is None):
        suffix = os.path.splitext(up.name)[1].lower() or ".mp4"
        bar = st.progress(0.0, text="Reading frames")

        def work():
            parts = []
            if not is_audio:
                parts.append(ex.extract_from_video_frames(
                    up.getvalue(), suffix, every, max_frames, progress=lambda f: bar.progress(f)))
            if is_audio or speech:
                bar.progress(1.0, text="Turning speech into text")
                try:
                    parts.append(ex.transcribe_media(up.getvalue(), suffix))
                except ValueError:
                    if is_audio:
                        raise
            return " ".join(parts)

        text = guarded(work)
        bar.empty()
        if text:
            run_analysis(text, f"{'Audio' if is_audio else 'Video'}: {up.name}")

# --- link
with tab_url:
    url = st.text_input("Link to a news article", placeholder="https://")
    st.caption("The article text is taken from the page. Pages behind a login or paywall will not work.")
    if st.button("Check link", type="primary", key="go_url"):
        if not url.strip():
            st.warning("Paste a link first.")
        else:
            with st.spinner("Fetching the page"):
                text = guarded(lambda: ex.extract_from_url(url))
            if text:
                run_analysis(text, f"Link: {url.strip()[:50]}")

# --- batch csv
with tab_csv:
    up = st.file_uploader("Upload a CSV with one article per row", type=["csv"], key="csv_up")
    if up:
        df = guarded(lambda: ex.read_csv_rows(up.getvalue()))
        if df is not None and len(df):
            cols = list(df.columns)
            col = st.selectbox("Column that holds the article text", cols,
                               index=cols.index(ex.guess_text_column(df)))
            limit = st.slider("Rows to check", 1, min(200, len(df)), min(25, len(df)))
            st.dataframe(df.head(5), hide_index=True)
            if st.button("Check all rows", type="primary", key="go_csv"):
                predictor = get_predictor()
                rows, bar = [], st.progress(0.0, text="Checking rows")
                for i, raw in enumerate(df[col].head(limit).astype(str)):
                    p = predictor.predict(raw)
                    rows.append({"Row": i + 1, "Text": raw[:90],
                                 "Prediction": "Potentially fake" if p["label"] == "FAKE" else "Likely real",
                                 "Confidence %": round(p["confidence"] * 100, 1),
                                 "Unlike training data": p["max_similarity"] < LOW_SIMILARITY,
                                 "_label": p["label"]})
                    bar.progress((i + 1) / limit, text=f"Checked {i + 1} of {limit}")
                bar.empty()
                out = pd.DataFrame(rows)
                match = None
                for c in df.columns:
                    if c.lower() == "label":
                        truth = df[c].head(limit).astype(str).str.upper().values
                        if set(truth) <= {"FAKE", "REAL"}:
                            match = float((truth == out["_label"].values).mean())
                st.session_state.batch = {"table": out.drop(columns="_label"), "match": match}

    if st.session_state.batch is not None:
        table = st.session_state.batch["table"]
        n_fake = int((table["Prediction"] == "Potentially fake").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows checked", len(table))
        m2.metric("Potentially fake", n_fake)
        m3.metric("Likely real", len(table) - n_fake)
        if st.session_state.batch["match"] is not None:
            st.info(f"The predictions match your label column in "
                    f"{st.session_state.batch['match']*100:.1f}% of rows.")
        st.dataframe(table, hide_index=True)
        st.download_button("Download results (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="batch_results.csv", mime="text/csv")

# --- shared result panel
if st.session_state.result:
    st.divider()
    show_result(st.session_state.result)
