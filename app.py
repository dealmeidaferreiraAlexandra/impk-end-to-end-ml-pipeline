# Developed by Alexandra de Almeida Ferreira

import json
import time
import io

import pandas as pd
import streamlit as st

# PDF
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from src.config import RAW_DATA_PATH, MODEL_VERSIONS
from src.models import load_saved_artifacts, predict_single_row, train_and_save_models

st.set_page_config(page_title="End-to-End ML Pipeline", layout="wide")

# =============================
# STYLE
# =============================
st.markdown("""
<style>
.stApp { background:#020617; color:#e2e8f0; }
.left-panel { border-right:1px solid #1f2231; padding-right:12px; }
.right-panel { background:#050a18; padding:20px; border-radius:16px; }
.stButton>button { width:100%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:10px; border:none; }

[data-testid="stFileUploaderDropzone"] {
    background:#020617;
    border:1px solid #1f2231;
    border-radius:12px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color:#6366f1;
    box-shadow:0 0 20px rgba(99,102,241,0.35);
}

[data-testid="stFileUploaderDropzone"] button {
    background:transparent;
    border:1px solid #1f2231;
    color:#e2e8f0;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {
    color:#e2e8f0;
}

.pipe {
    border:1px solid #1f2231;
    border-radius:12px;
    padding:12px;
    text-align:center;
    background:#020617;
}

.active {
    border:1px solid #6366f1;
    box-shadow:0 0 20px rgba(99,102,241,0.6);
}

.card {
    border:1px solid #1f2231;
    border-radius:14px;
    padding:16px;
    margin-top:20px;
    background:#020617;
}

.footer {
    text-align:center;
    opacity:0.6;
    margin-top:30px;
}

.badge {
    display:inline-block;
    padding:6px 10px;
    border-radius:8px;
    background:#16a34a;
    color:white;
    font-size:12px;
    margin-bottom:10px;
}

.system-status {
    background:transparent;
    border:none;
    padding:0;
    margin:4px 0;
    font-weight:600;
}

.system-ready {
    color:#22c55e;
}

.system-waiting {
    color:#ef4444;
}

.model-card {
    min-height:280px;
}

.model-card p {
    margin:5px 0;
    font-size:13px;
}

.glow-green {
    border-color:#22c55e;
    box-shadow:0 0 18px rgba(34,197,94,0.55);
}

.glow-yellow {
    border-color:#eab308;
    box-shadow:0 0 18px rgba(234,179,8,0.45);
}

.glow-red {
    border-color:#ef4444;
    box-shadow:0 0 18px rgba(239,68,68,0.45);
}

.legend {
    display:flex;
    justify-content:flex-end;
    gap:12px;
    flex-wrap:wrap;
    margin:12px 0 4px 0;
}

.legend span {
    display:inline-flex;
    align-items:center;
    gap:6px;
    font-size:12px;
    opacity:0.82;
}

.legend i {
    display:inline-block;
    width:10px;
    height:10px;
    border-radius:50%;
}

.legend .green { background:#22c55e; }
.legend .yellow { background:#eab308; }
.legend .red { background:#ef4444; }

.results-header {
    display:flex;
    align-items:center;
    gap:12px;
    flex-wrap:wrap;
    margin:0 0 10px 0;
}

.results-header h2 {
    margin:0;
    line-height:1;
}

.results-header .badge {
    margin:0;
    line-height:1;
    display:inline-flex;
    align-items:center;
    height:24px;
}

.outcome-lived {
    color:#22c55e;
    font-weight:700;
}

.outcome-died {
    color:#ef4444;
    font-weight:700;
}

.country-small {
    font-size:11px;
    opacity:0.65;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🧠 End-to-End ML Pipeline")
st.caption("Titanic dataset | Data cleaning + feature engineering + model comparison")

# =============================
# STATE
# =============================
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "upload_key" not in st.session_state:
    st.session_state.upload_key = "upload_0"
if "source_df" not in st.session_state:
    st.session_state.source_df = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None
if "pipeline_started" not in st.session_state:
    st.session_state.pipeline_started = False
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# =============================
# LOAD SAVED
# =============================
saved_models, saved_metrics, saved_best = load_saved_artifacts()
if saved_models and not st.session_state.models:
    st.session_state.models = saved_models
if saved_metrics is not None and st.session_state.metrics_df is None:
    st.session_state.metrics_df = saved_metrics
if saved_best is not None and st.session_state.best_model is None:
    st.session_state.best_model = saved_best

# =============================
# PIPELINE
# =============================
def render_pipeline(stage: str):
    p1, p2, p3, p4, p5 = st.columns(5)

    def pipe(col, icon, title, desc, key):
        active = stage == key
        with col:
            st.markdown(f"""
                <div class="pipe {'active' if active else ''}">
                    {icon}<br>
                    <b>{title}</b><br>
                    <span style="opacity:0.6;font-size:11px;">{desc}</span>
                </div>
            """, unsafe_allow_html=True)

    pipe(p1, "📥", "LOAD", "Upload CSV", "upload")
    pipe(p2, "🧹", "CLEAN", "Fill missing data", "clean")
    pipe(p3, "🧩", "FEATURE", "Engineer features", "feature")
    pipe(p4, "🧠", "TRAIN", "Fit 3 models", "train")
    pipe(p5, "📊", "COMPARE", "Compare results", "compare")

# =============================
# PDF TABLE
# =============================
def df_to_table(df):
    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#6366f1")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))
    return table

def model_glow_class(model_name):
    model_order = st.session_state.metrics_df["Model"].tolist() if st.session_state.metrics_df is not None else []
    if model_name == st.session_state.best_model or (model_order and model_name == model_order[0]):
        return "glow-green"
    if len(model_order) > 1 and model_name == model_order[1]:
        return "glow-yellow"
    return "glow-red"

def metric_glow_class(index):
    if index == 0:
        return "glow-green"
    if index == 1:
        return "glow-yellow"
    return "glow-red"

def render_color_legend():
    st.markdown(
        """
        <div class="legend">
            <span><i class="green"></i>Green: best model</span>
            <span><i class="yellow"></i>Yellow: second best model</span>
            <span><i class="red"></i>Red: lowest model</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(model_name, metrics, glow_class=""):
    rows = "".join(
        f"<p><b>{key}:</b> {value}</p>"
        for key, value in metrics.items()
        if key != "Model"
    )
    st.markdown(
        f"""
        <div class="card model-card {glow_class}">
            <h4>{model_name}</h4>
            {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

def prediction_report(predictions):
    return {
        model_name: {
            "label": pred["label"],
            "survival_prob": pred["survival_prob"],
            "uncertainty": pred["uncertainty"],
        }
        for model_name, pred in ordered_predictions(predictions)
    }

def ordered_predictions(predictions):
    if st.session_state.metrics_df is not None:
        model_order = st.session_state.metrics_df["Model"].tolist()
        return [(model_name, predictions[model_name]) for model_name in model_order if model_name in predictions]
    return list(predictions.items())

TICKET_CLASS_LABELS = {
    "First": 1,
    "Second": 2,
    "Third": 3,
}

EMBARKED_LABELS = {
    "Southampton (England)": "S",
    "Cherbourg (France)": "C",
    "Queenstown (Ireland)": "Q",
}

def label_for_value(mapping, value, fallback):
    for label, mapped_value in mapping.items():
        if mapped_value == value:
            return label
    return fallback

def survival_label(value):
    if pd.isna(value):
        return "Unknown"
    return "Lived" if int(value) == 1 else "Died"

def prediction_outcome_label(pred):
    return "Lived" if pred["label"] == "Survived" else "Died"

def outcome_html(label):
    css_class = "outcome-lived" if label == "Lived" else "outcome-died"
    return f'<span class="{css_class}">{label}</span>'

def display_passenger_df(df):
    display_df = df.copy()

    if "Pclass" in display_df.columns:
        display_df["Ticket Class"] = display_df["Pclass"].apply(
            lambda value: label_for_value(TICKET_CLASS_LABELS, int(value), value) if pd.notna(value) else "Unknown"
        )
        display_df = display_df.drop(columns=["Pclass"])

    if "Embarked" in display_df.columns:
        display_df["Embarked"] = display_df["Embarked"].apply(
            lambda value: label_for_value(EMBARKED_LABELS, str(value).upper(), value) if pd.notna(value) else "Unknown"
        )

    if "Survived" in display_df.columns:
        display_df["Outcome"] = display_df["Survived"].apply(survival_label)
        display_df = display_df.drop(columns=["Survived"])

    ordered_cols = [
        "PassengerId",
        "Outcome",
        "Ticket Class",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    ]
    return display_df[[col for col in ordered_cols if col in display_df.columns]]

def passenger_table_html(df):
    display_df = display_passenger_df(df)
    if "Outcome" in display_df.columns:
        display_df["Outcome"] = display_df["Outcome"].apply(
            lambda value: outcome_html(value) if value in ["Lived", "Died"] else value
        )
    return display_df.to_html(index=False, escape=False)

def fare_options_from_df(df):
    fares = pd.to_numeric(df["Fare"], errors="coerce").dropna() if "Fare" in df.columns else pd.Series(dtype=float)
    if fares.empty:
        return {"Low fare": 0.0, "Medium fare": 15.0, "High fare": 50.0}

    values = fares.quantile([0.25, 0.5, 0.75]).round(2).tolist()
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(float(value))

    labels = ["Low fare", "Medium fare", "High fare"]
    return {label: value for label, value in zip(labels, unique_values)}

def nearest_label(options, value):
    return min(options, key=lambda label: abs(options[label] - value))

if REPORTLAB_AVAILABLE:
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            page_count = len(self._saved_page_states)
            for page_state in self._saved_page_states:
                self.__dict__.update(page_state)
                self.setFont("Helvetica", 9)
                self.drawRightString(780, 18, f"{self._pageNumber}/{page_count}")
                super().showPage()
            super().save()

def predictions_to_pdf_table(predictions):
    styles = getSampleStyleSheet()
    headers = []
    details = []
    backgrounds = []

    for model_name, pred in ordered_predictions(predictions):
        survival = pred["survival_prob"]
        uncertainty = pred["uncertainty"]
        outcome = prediction_outcome_label(pred)
        outcome_color = "green" if outcome == "Lived" else "red"
        headers.append(Paragraph(f"<b>{model_name}</b>", styles["Normal"]))
        details.append(
            Paragraph(
                f"""
                Outcome: <font color="{outcome_color}"><b>{outcome}</b></font><br/>
                Survival probability: {survival*100:.2f}%<br/>
                Uncertainty: {uncertainty*100:.2f}%<br/>
                Non-survival probability: {(1-survival)*100:.2f}%<br/>
                Confidence: {(1-uncertainty)*100:.2f}%<br/>
                Risk score: {(survival*uncertainty)*100:.2f}%<br/>
                Decision threshold: 50%<br/>
                Model version: {MODEL_VERSIONS.get(model_name, "v1")}
                """,
                styles["Normal"],
            )
        )

        glow_class = model_glow_class(model_name)
        if glow_class == "glow-green":
            backgrounds.append(colors.HexColor("#dcfce7"))
        elif glow_class == "glow-yellow":
            backgrounds.append(colors.HexColor("#fef9c3"))
        else:
            backgrounds.append(colors.HexColor("#fee2e2"))

    table = Table([headers, details], colWidths=[220] * len(headers))
    style = [
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#6366f1")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('ALIGN',(0,0),(-1,0),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]

    for col_idx, background in enumerate(backgrounds):
        style.append(('BACKGROUND',(col_idx,1),(col_idx,1),background))

    table.setStyle(TableStyle(style))
    return table

# =============================
# LAYOUT
# =============================
left, right = st.columns([0.22, 0.78])

# =============================
# LEFT PANEL
# =============================
with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Titanic train.csv", type=["csv"], key=st.session_state.upload_key)

    if uploaded is not None:
        st.session_state.source_df = pd.read_csv(uploaded)
        st.session_state.dataset_name = "Uploaded dataset"
        # clear any cached preview when a new dataset is uploaded
        st.session_state.pop("random_preview", None)
        st.session_state.pop("random_preview_dataset", None)
        st.session_state.pop("what_if", None)
        st.session_state.pop("what_if_base_idx", None)

    if st.session_state.source_df is None and RAW_DATA_PATH.exists():
        st.session_state.source_df = pd.read_csv(RAW_DATA_PATH)
        st.session_state.dataset_name = "Titanic demo dataset"
        # clear any cached preview when demo dataset is loaded
        st.session_state.pop("random_preview", None)
        st.session_state.pop("random_preview_dataset", None)
        st.session_state.pop("what_if", None)
        st.session_state.pop("what_if_base_idx", None)

    df = st.session_state.source_df

    train_clicked = st.button("Run pipeline")

    st.subheader("System")
    if df is not None:
        st.markdown(
            f'<div class="system-status system-ready">🟢 Dataset ready — {len(df)} rows</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Columns: {len(df.columns)}")
    else:
        st.markdown(
            '<div class="system-status system-waiting">🔴 Waiting for Titanic CSV</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# RIGHT PANEL
# =============================
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    render_pipeline(st.session_state.stage)

    if not st.session_state.pipeline_started:
        st.markdown("""
        <div class="card">
            <h3>🚀 Start</h3>
            Run the pipeline using the built-in Titanic dataset<br><br>
            or upload your own CSV to explore the full machine learning workflow.<br><br>
            Train models, compare performance, and test predictions interactively.
        </div>
        """, unsafe_allow_html=True)

    if train_clicked and df is not None:
        st.session_state.pipeline_started = True
        st.session_state.stage = "train"
        st.session_state.metrics_df = None
        st.session_state.pdf_buffer = None
        st.rerun()

    if st.session_state.pipeline_started and st.session_state.stage == "train":
        with st.spinner("Training models..."):
            models, metrics_df, best_model = train_and_save_models(df)
            st.session_state.models = models
            st.session_state.metrics_df = metrics_df
            st.session_state.best_model = best_model
            st.session_state.stage = "compare"
            sample = df.iloc[[0]]
            preds = predict_single_row(st.session_state.models, sample)

            # 🔥  PDF 
            if REPORTLAB_AVAILABLE:
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
                styles = getSampleStyleSheet()

                elements = [
                    Paragraph("ML Pipeline Report", styles["Title"]),
                    Spacer(1, 12),

                    Paragraph(f"Dataset: {st.session_state.dataset_name}", styles["Normal"]),
                    Paragraph(f"Rows: {len(df)}", styles["Normal"]),
                    Paragraph(f"Best model: {st.session_state.best_model}", styles["Normal"]),

                    Spacer(1, 12),
                    Paragraph("Metrics", styles["Heading2"]),
                    df_to_table(st.session_state.metrics_df),

                    Spacer(1, 12),
                    Paragraph("Sample analyzed", styles["Heading2"]),
                    df_to_table(sample),

                    Spacer(1, 12),
                    Paragraph("Predictions", styles["Heading2"]),
                ]

                #  outputs 2 PDF models
                for model_name, pred in ordered_predictions(preds):
                    survival = pred["survival_prob"]
                    uncertainty = pred["uncertainty"]
                    outcome = prediction_outcome_label(pred)

                    text = f"""
                    {model_name}<br/>
                    Outcome: {outcome}<br/>
                    Survival: {survival*100:.2f}%<br/>
                    Uncertainty: {uncertainty*100:.2f}%<br/>
                    Confidence: {(1-uncertainty)*100:.2f}%<br/>
                    Risk score: {(survival*uncertainty)*100:.2f}%<br/>
                    """

                    elements.append(Paragraph(text, styles["Normal"]))
                    elements.append(Spacer(1, 10))

                # footer
                elements.append(Spacer(1, 20))
                elements.append(Paragraph("Developed by Alexandra de Almeida Ferreira", styles["Normal"]))
                elements.append(Paragraph('<link href="https://github.com/dealmeidaferreiraAlexandra">https://github.com/dealmeidaferreiraAlexandra</link>', styles["Normal"]))
                elements.append(Paragraph('<link href="https://www.linkedin.com/in/dealmeidaferreira">https://www.linkedin.com/in/dealmeidaferreira</link>', styles["Normal"]))

                doc.build(elements, canvasmaker=NumberedCanvas)
                st.session_state.pdf_buffer = buffer.getvalue()
        st.rerun()

    # =============================
    # RESULTS
    # =============================
    if st.session_state.pipeline_started and st.session_state.metrics_df is not None:
        dataset_badge = ""
        if st.session_state.dataset_name == "Titanic demo dataset":
            dataset_badge = '<span class="badge">Titanic demo dataset loaded</span>'
        elif st.session_state.dataset_name:
            dataset_badge = f'<span class="badge">{st.session_state.dataset_name} loaded</span>'

        st.markdown(
            f"""
            <div class="results-header">
                <h2>🧠 Results</h2>
                {dataset_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_color_legend()
    
        # preview 5 random passengers
        if df is not None:
            st.markdown("### 👥 Random passenger preview")
            # persist the preview in session_state so other interactions don't change it
            if "random_preview" not in st.session_state or st.session_state.get("random_preview_dataset") != st.session_state.dataset_name:
                st.session_state.random_preview = df.sample(min(5, len(df)), random_state=None).copy(deep=True)
                st.session_state.random_preview_dataset = st.session_state.dataset_name
            random_preview = st.session_state.random_preview
            st.markdown(
                f"<div class='card'>{passenger_table_html(random_preview)}</div>",
                unsafe_allow_html=True,
            )
    
        # test passenger section
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.markdown("### 🔍 Test a passenger")

        select_col, random_col = st.columns([0.28, 0.72])
        with select_col:
            row_idx = st.number_input("Select passenger index", 0, max(len(df) - 1, 0), 0)

        with random_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🎲 Random passenger"):
                # pick a truly random row from the dataframe and convert its label to a positional index
                idx_label = df.sample(n=1).index[0]
                st.session_state.random_idx = int(df.index.get_loc(idx_label))
    
        if "random_idx" in st.session_state:
            row_idx = st.session_state.random_idx
    
        sample = df.iloc[[int(row_idx)]].copy(deep=True)
    
        st.markdown(f"<div class='card'>{passenger_table_html(sample)}</div>", unsafe_allow_html=True)
    
        preds = predict_single_row(st.session_state.models, sample)
    
        pred_cols = st.columns(3)
        for col, (model_name, pred) in zip(pred_cols, ordered_predictions(preds)):
            with col:
                survival = pred["survival_prob"]
                uncertainty = pred["uncertainty"]
                glow_class = model_glow_class(model_name)
    
                label_human = prediction_outcome_label(pred)
    
                st.markdown(
                    f"""
                    <div class="card model-card {glow_class}">
                        <h4>{model_name}</h4>
                        <p><b>Outcome:</b> {outcome_html(label_human)}</p>
                        <p><b>Survival probability:</b> {survival*100:.2f}%</p>
                        <p><b>Uncertainty:</b> {uncertainty*100:.2f}%</p>
                        <p><b>Non-survival probability:</b> {(1-survival)*100:.2f}%</p>
                        <p><b>Confidence:</b> {(1-uncertainty)*100:.2f}%</p>
                        <p><b>Risk score:</b> {(survival*uncertainty)*100:.2f}%</p>
                        <p><b>Decision threshold:</b> 50%</p>
                        <p><b>Model version:</b> {MODEL_VERSIONS.get(model_name, "v1")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.markdown("### 🧪 What-if analysis")

        # Initialize or reuse a persistent what_if stored in session_state so edits
        # to the what-if scenario do not get overwritten when the sample changes.
        # Initialize what_if only once per session (until user resets or dataset changes).
        # Start it from the current sample but remove identifying fields like PassengerId.
        if "what_if" not in st.session_state:
            new_what_if = sample.copy(deep=True)
            new_what_if = new_what_if.drop(columns=["PassengerId"], errors="ignore")
            st.session_state.what_if = new_what_if

        what_if = st.session_state.what_if.copy(deep=True)
        sex_options = ["male", "female"]
        fare_options = fare_options_from_df(df)

        current_pclass = int(what_if["Pclass"].iloc[0]) if "Pclass" in what_if.columns and pd.notna(what_if["Pclass"].iloc[0]) else 3
        current_sex = str(what_if["Sex"].iloc[0]).lower() if "Sex" in what_if.columns and pd.notna(what_if["Sex"].iloc[0]) else "male"
        current_age = float(what_if["Age"].iloc[0]) if "Age" in what_if.columns and pd.notna(what_if["Age"].iloc[0]) else 30.0
        current_fare = float(what_if["Fare"].iloc[0]) if "Fare" in what_if.columns and pd.notna(what_if["Fare"].iloc[0]) else 0.0
        current_sibsp = int(what_if["SibSp"].iloc[0]) if "SibSp" in what_if.columns and pd.notna(what_if["SibSp"].iloc[0]) else 0
        current_parch = int(what_if["Parch"].iloc[0]) if "Parch" in what_if.columns and pd.notna(what_if["Parch"].iloc[0]) else 0
        current_embarked = str(what_if["Embarked"].iloc[0]).upper() if "Embarked" in what_if.columns and pd.notna(what_if["Embarked"].iloc[0]) else "S"
        current_cabin = str(what_if["Cabin"].iloc[0]) if "Cabin" in what_if.columns and pd.notna(what_if["Cabin"].iloc[0]) else ""

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            ticket_class_label = st.selectbox(
                "Ticket Class",
                list(TICKET_CLASS_LABELS.keys()),
                index=list(TICKET_CLASS_LABELS.values()).index(current_pclass) if current_pclass in TICKET_CLASS_LABELS.values() else 2,
            )
            what_if_pclass = TICKET_CLASS_LABELS[ticket_class_label]
        with w2:
            what_if_sex = st.selectbox(
                "Sex",
                sex_options,
                index=sex_options.index(current_sex) if current_sex in sex_options else 0,
            )
        with w3:
            what_if_age = st.number_input("Age", 0.0, 100.0, current_age)
        with w4:
            fare_label = st.selectbox(
                "Fare",
                list(fare_options.keys()),
                index=list(fare_options.keys()).index(nearest_label(fare_options, current_fare)),
                format_func=lambda label: f"{label}: {fare_options[label]:.2f}",
            )
            what_if_fare = fare_options[fare_label]

        w5, w6, w7, w8 = st.columns(4)
        with w5:
            what_if_sibsp = st.number_input("Siblings/spouses", 0, 10, current_sibsp)
        with w6:
            what_if_parch = st.number_input("Parents/children", 0, 10, current_parch)
        with w7:
            st.markdown('Embarked <span class="country-small">(city, country)</span>', unsafe_allow_html=True)
            embarked_label = st.selectbox(
                "Embarked",
                list(EMBARKED_LABELS.keys()),
                index=list(EMBARKED_LABELS.values()).index(current_embarked) if current_embarked in EMBARKED_LABELS.values() else 0,
                label_visibility="collapsed",
            )
            what_if_embarked = EMBARKED_LABELS[embarked_label]
        with w8:
            what_if_has_cabin = st.checkbox("Cabin known", value=bool(current_cabin))

        # (Reset button removed by request)

        what_if.loc[:, "Pclass"] = what_if_pclass
        what_if.loc[:, "Sex"] = what_if_sex
        what_if.loc[:, "Age"] = what_if_age
        what_if.loc[:, "Fare"] = what_if_fare
        what_if.loc[:, "SibSp"] = what_if_sibsp
        what_if.loc[:, "Parch"] = what_if_parch
        what_if.loc[:, "Embarked"] = what_if_embarked
        what_if.loc[:, "Cabin"] = current_cabin if what_if_has_cabin and current_cabin else ("WHATIF" if what_if_has_cabin else pd.NA)
        # clear Name for what-if so it doesn't persist from sample or previews
        what_if.loc[:, "Name"] = pd.NA

        # persist what-if edits so they don't get lost on rerun
        st.session_state.what_if = what_if.copy(deep=True)
        # display the what-if without any Name column to avoid leaking or persisting names
        what_if_ui = what_if.copy(deep=True)
        what_if_ui = what_if_ui.drop(columns=["Name"], errors="ignore")
        st.markdown(f"<div class='card'>{passenger_table_html(what_if_ui)}</div>", unsafe_allow_html=True)

        what_if_preds = predict_single_row(st.session_state.models, what_if)

        what_if_cols = st.columns(3)
        for col, (model_name, pred) in zip(what_if_cols, ordered_predictions(what_if_preds)):
            with col:
                survival = pred["survival_prob"]
                uncertainty = pred["uncertainty"]
                glow_class = model_glow_class(model_name)
                label_human = prediction_outcome_label(pred)

                st.markdown(
                    f"""
                    <div class="card model-card {glow_class}">
                        <h4>{model_name}</h4>
                        <p><b>Outcome:</b> {outcome_html(label_human)}</p>
                        <p><b>Survival probability:</b> {survival*100:.2f}%</p>
                        <p><b>Uncertainty:</b> {uncertainty*100:.2f}%</p>
                        <p><b>Non-survival probability:</b> {(1-survival)*100:.2f}%</p>
                        <p><b>Confidence:</b> {(1-uncertainty)*100:.2f}%</p>
                        <p><b>Risk score:</b> {(survival*uncertainty)*100:.2f}%</p>
                        <p><b>Decision threshold:</b> 50%</p>
                        <p><b>Model version:</b> {MODEL_VERSIONS.get(model_name, "v1")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # metrics after passenger section
        metrics_df = st.session_state.metrics_df.copy()
    
        st.markdown("### 📊 Model performance")
    
        cols = st.columns(3)
        for index, (col, (_, row)) in enumerate(zip(cols, metrics_df.iterrows())):
            with col:
                metric_card(row["Model"], row.to_dict(), metric_glow_class(index))
    
        # spacing before buttons
        st.markdown("<div style='margin-top:25px'></div>", unsafe_allow_html=True)
        sample_display = display_passenger_df(sample)
        # For exports and UI, remove Name from what-if to avoid persisting it
        what_if_display = display_passenger_df(what_if.drop(columns=["Name"], errors="ignore"))
        sample_export = sample_display.astype(object).where(pd.notna(sample_display), None)
        what_if_export = what_if_display.astype(object).where(pd.notna(what_if_display), None)
        what_if_export.loc[:, "TicketClassLabel"] = ticket_class_label
        what_if_export.loc[:, "FareLabel"] = fare_label
        what_if_export.loc[:, "EmbarkedLabel"] = embarked_label
    
        # JSON REPORT
        report = {
            "author": "Alexandra de Almeida Ferreira",
            "project": "End-to-End ML Pipeline",
            "dataset": st.session_state.dataset_name,
            "dataset_rows": int(len(df)) if df is not None else None,
            "best_model": st.session_state.best_model,
            "model_versions": MODEL_VERSIONS,
            "metrics": st.session_state.metrics_df.to_dict(orient="records"),
            "sample_analyzed": sample_export.to_dict(orient="records"),
            "predictions": prediction_report(preds),
            "what_if": {
                "scenario": what_if_export.to_dict(orient="records"),
                "predictions": prediction_report(what_if_preds),
            },
            "links": {
                "github": "https://github.com/dealmeidaferreiraAlexandra",
                "linkedin": "https://www.linkedin.com/in/dealmeidaferreira",
            },
        }
    
        # PDF
        if REPORTLAB_AVAILABLE:
            with st.spinner("Generating PDF..."):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
                styles = getSampleStyleSheet()
    
                elements = [
                    Paragraph("ML Pipeline Report", styles["Title"]),
                    Spacer(1, 12),
                    Paragraph(f"Dataset: {st.session_state.dataset_name}", styles["Normal"]),
                    Paragraph(f"Rows: {len(df)}", styles["Normal"]),
                    Paragraph(f"Best model: {st.session_state.best_model}", styles["Normal"]),
                    Spacer(1, 12),
                    Paragraph("Metrics", styles["Heading2"]),
                    df_to_table(st.session_state.metrics_df),
                    Spacer(1, 12),
                    Paragraph("Sample analyzed", styles["Heading2"]),
                    df_to_table(sample_export),
                    Spacer(1, 12),
                    Paragraph("Color legend", styles["Heading2"]),
                    Paragraph("Green: best model | Yellow: second best model | Red: lowest model", styles["Normal"]),
                    Spacer(1, 12),
                    Paragraph("Predictions", styles["Heading2"]),
                    predictions_to_pdf_table(preds),
                ]

                elements += [
                    Spacer(1, 12),
                    Paragraph("What-if scenario", styles["Heading2"]),
                    df_to_table(what_if_export),
                    Spacer(1, 12),
                    Paragraph("What-if predictions", styles["Heading2"]),
                    predictions_to_pdf_table(what_if_preds),
                ]

                elements += [
                    Spacer(1, 20),
                    Paragraph("Developed by Alexandra de Almeida Ferreira", styles["Normal"]),
                    Paragraph('<link href="https://github.com/dealmeidaferreiraAlexandra">GitHub</link>', styles["Normal"]),
                    Paragraph('<link href="https://www.linkedin.com/in/dealmeidaferreira">LinkedIn</link>', styles["Normal"]),
                ]

                doc.build(elements, canvasmaker=NumberedCanvas)
                st.session_state.pdf_buffer = buffer.getvalue()
    
        # BUTTONS
        c1, c2, c3 = st.columns(3)
    
        with c1:
            st.download_button(
                "⬇️ JSON",
                json.dumps(report, indent=2, ensure_ascii=False),
                "report.json",
                use_container_width=True,
            )
    
        with c2:
            if REPORTLAB_AVAILABLE and st.session_state.pdf_buffer:
                st.download_button(
                    "📄 PDF Report",
                    st.session_state.pdf_buffer,
                    "report.pdf",
                    use_container_width=True,
                )
    
        with c3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.session_state.stage = "upload"
                st.session_state.upload_key = f"upload_{time.time()}"
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("""
<div class='footer'>
Developed by <b>Alexandra de Almeida Ferreira</b><br><br>
🔗 <a href="https://github.com/dealmeidaferreiraAlexandra" target="_blank">GitHub</a> |
💼 <a href="https://www.linkedin.com/in/dealmeidaferreira" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
