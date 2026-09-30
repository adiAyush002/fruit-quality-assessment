import pandas as pd
import streamlit as st

from src.config import TASKS
from src.prediction import (PredictionError, SUPPORTED_EXTENSIONS, decode_image,
                            load_models, predict_image)
from src.quality_rules import assess_quality

st.set_page_config(page_title="Fruit Quality Assessment", page_icon="🍎", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return load_models()


def pct(x):
    return f"{x * 100:.1f}%"


def show_results(results):
    st.markdown("#### Deep learning predictions")
    cols = st.columns(len(TASKS))
    for col, task in zip(cols, TASKS):
        r = results[task]
        with col:
            st.metric(TASKS[task]["title"], r["label"].capitalize())
            st.progress(float(r["confidence"]))
            st.caption(f"Confidence: {pct(r['confidence'])}")

    st.divider()
    st.markdown("#### Rule-based quality assessment")
    st.caption("Not a deep-learning prediction. It is produced by fixed rules that combine the "
               "ripeness and defect predictions (see src/quality_rules.py).")
    quality = assess_quality(results["defect"]["label"], results["ripeness"]["label"],
                             {t: results[t]["confidence"] for t in results})
    box = {"Good": st.success, "Average": st.warning, "Poor": st.error}.get(quality["quality"], st.info)
    box(f"**Overall quality: {quality['quality']}**")
    st.write(quality["explanation"])
    for w in quality["warnings"]:
        st.warning(w)

    with st.expander("Show class probabilities"):
        for task in TASKS:
            st.write(f"**{TASKS[task]['title']}**")
            st.bar_chart(pd.DataFrame({"probability": results[task]["probabilities"]}))


st.title("Fruit Quality, Ripeness & Defect Assessment")
st.caption("Upload a fruit photo. Three deep-learning models predict the fruit type, ripeness and visible defect.")

with st.sidebar:
    st.header("About")
    st.write("Models: MobileNetV2 transfer learning (one model per task).")
    st.write("Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS))
    st.header("Limitations")
    st.write("This tool only assesses **visible characteristics** in the image. It cannot judge internal "
             "quality, taste, nutrition, pesticide residue or hidden defects, and it does not replace "
             "professional food-quality inspection.")

try:
    models = get_models()
except PredictionError as e:
    st.error(str(e))
    st.info("Train the models first:\n\n`python -m src.train_fruit`\n\n`python -m src.train_ripeness`\n\n"
            "`python -m src.train_defect`")
    st.stop()

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("1. Upload image")
    uploaded = st.file_uploader("Choose a fruit image",
                                type=[e.strip(".") for e in SUPPORTED_EXTENSIONS])
    image = None
    if uploaded is not None:
        try:
            image = decode_image(uploaded.getvalue(), uploaded.name)
            st.image(image, caption=uploaded.name, width=380)
        except PredictionError as e:
            st.error(str(e))
    run = st.button("Run prediction", type="primary")

with right:
    st.subheader("2. Results")
    if not run:
        st.info("Upload an image and click **Run prediction**.")
    elif uploaded is None:
        st.warning("No image uploaded. Please upload an image first.")
    elif image is None:
        st.error("This file cannot be analysed. Please upload a valid image.")
    else:
        try:
            with st.spinner("Analysing image..."):
                results = predict_image(image, models)
            show_results(results)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
