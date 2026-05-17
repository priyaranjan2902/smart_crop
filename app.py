import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import litellm
import numpy as np
import cv2
from ultralytics import YOLO
import os

# =============================
# 🔹 CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['GROQ_API_KEY'] = "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"

# =============================
# 🔹 LOAD CLASSES
# =============================
with open("classes.json") as f:
    classes = json.load(f)

plant_names = sorted(list(set([c.split("___")[0] for c in classes])))

# =============================
# 🔹 LOAD CLASSIFICATION MODEL
# =============================
@st.cache_resource
def load_classifier():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

classifier = load_classifier()

# =============================
# 🔹 LOAD SEGMENTATION MODEL
# =============================
@st.cache_resource
def load_segmenter():
    return YOLO("best.pt")

segmenter = load_segmenter()

# =============================
# 🔹 TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])
# =============================
# 🔹 CLASSIFICATION
# =============================
def predict(image, selected_plant):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    filtered = []
    for i, cls in enumerate(classes):
        if cls.startswith(selected_plant):
            filtered.append((cls, probs[i]))

    # Normalize within selected plant classes
    total_prob = sum([x[1] for x in filtered])

    normalized = []
    for cls, prob in filtered:
        normalized_prob = prob / total_prob if total_prob > 0 else prob
        normalized.append((cls, normalized_prob))

    normalized.sort(key=lambda x: x[1], reverse=True)

    best_class, confidence = normalized[0]

    plant, disease = best_class.split("___")

    return plant, disease, confidence

# =============================
# 🔹 SEGMENTATION
# =============================
def segment_image(image):
    img_np = np.array(image)

    results = segmenter(img_np)

    combined_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

    for r in results:
        if r.masks is not None:
            for m in r.masks.data:
                mask = m.cpu().numpy()

                # Resize mask
                mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

                # Balanced threshold
                mask = (mask > 0.15).astype(np.uint8)

                # Combine masks
                combined_mask = np.maximum(combined_mask, mask)

    # Slight expansion for better disease visibility
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    infected_percent = (np.sum(combined_mask) / combined_mask.size) * 100

    # Red overlay
    colored_mask = np.zeros_like(img_np)
    colored_mask[:, :, 2] = combined_mask * 255

    overlay = cv2.addWeighted(img_np, 0.72, colored_mask, 0.55, 0)

    return overlay, infected_percent

# =============================
# 🔹 LLM (Groq via LiteLLM)
# =============================
def get_disease_info(plant, disease):
    prompt = f"""
    Plant: {plant}
    Disease: {disease}

    Explain:
    - What is this disease?
    - Causes
    - Symptoms
    - Prevention & treatment

    Keep it simple and useful for farmers.
    """

    response = litellm.completion(
        model="groq/openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]

# =============================
# 🔹 UI
# =============================
# =============================
# 🔹 PAGE CONFIG
# =============================
st.set_page_config(
    page_title="🌿 SmartCrop AI Advisor",
    page_icon="🌱",
    layout="wide"
)

# =============================
# 🔹 STYLES
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap');

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarCollapsedControl"],
header[data-testid="stHeader"],
#MainMenu,
.stDeployButton,
footer { display: none !important; }

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    scroll-behavior: smooth;
}

/* ── Background ── */
.main {
    background: #f5f3ff;
}

section[data-testid="stMain"] {
    background: #f5f3ff;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── LANDING PAGE ── */
.landing-wrapper {
    min-height: 82vh;
    background: linear-gradient(160deg, #062e2a 0%, #0a4a43 45%, #0f7a6e 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem 2rem 0.5rem;
    position: relative;
    overflow: hidden;
}

.landing-wrapper::before {
    content: "";
    position: absolute;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(56,239,125,0.12) 0%, transparent 70%);
    top: -100px; right: -100px;
    border-radius: 50%;
}

.landing-wrapper::after {
    content: "";
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(15,155,142,0.15) 0%, transparent 70%);
    bottom: -80px; left: -80px;
    border-radius: 50%;
}

.landing-badge {
    display: inline-block;
    background: rgba(56,239,125,0.15);
    border: 1px solid rgba(56,239,125,0.35);
    color: #38ef7d;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    margin-bottom: 1.8rem;
    z-index: 2;
    position: relative;
}

.landing-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 1.2rem;
    z-index: 2;
    position: relative;
}

.landing-title span {
    background: linear-gradient(90deg, #38ef7d, #11998e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.landing-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.88);
    max-width: 520px;
    line-height: 1.8;
    margin-bottom: 2.2rem;
    z-index: 2;
    position: relative;
}

.landing-features {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    margin-bottom: 2.2rem;
    flex-wrap: wrap;
    z-index: 2;
    position: relative;
}

.feat-chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.85);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    backdrop-filter: blur(6px);
}

/* ── LANDING BUTTON ── */
div[data-testid="stButton"].landing-btn > button,
.landing-btn .stButton > button {
    background: linear-gradient(135deg, #38ef7d, #11998e) !important;
    color: #062e2a !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.9rem 3rem !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: 0 8px 30px rgba(56,239,125,0.35) !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
}

div[data-testid="stButton"].landing-btn > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 40px rgba(56,239,125,0.5) !important;
}

/* ── PRIMARY BUTTON (Detect Disease) ── */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    background: linear-gradient(135deg, #38ef7d, #11998e);
    color: #062e2a;
    font-family: 'DM Mono', monospace;
    font-weight: 700;
    border: none;
    padding: 0.75rem 1.5rem;
    font-size: 0.82rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 4px 18px rgba(56,239,125,0.3);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(56,239,125,0.45);
}

.stButton > button:active { transform: scale(0.98); }

/* ── SECONDARY BUTTON (Show Disease Area) ── */
.stButton:nth-of-type(2) > button {
    background: transparent;
    color: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: none;
}

.stButton:nth-of-type(2) > button:hover {
    background: rgba(124,58,237,0.08);
    box-shadow: 0 4px 16px rgba(124,58,237,0.2);
    transform: translateY(-2px);
}

/* ── NAVBAR / TOP BAR ── */
.topbar {
    background: linear-gradient(90deg, #1e1b4b, #312e81);
    padding: 1.1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 24px rgba(30,27,75,0.3);
    margin-bottom: 0;
}

.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.topbar-logo span { color: #fbbf24; }

.topbar-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #fbbf24;
    background: rgba(251,191,36,0.12);
    border: 1px solid rgba(251,191,36,0.3);
    padding: 0.28rem 0.9rem;
    border-radius: 4px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── SECTION CARDS ── */
.panel {
    background: white;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 24px rgba(79,70,229,0.09), 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 1.4rem;
    border: 1px solid #ede9fe;
    position: relative;
    overflow: hidden;
}

.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7c3aed, #4f46e5, #fbbf24);
}

.panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── RESULT GRID ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
    margin: 1.2rem 0;
}

.result-tile {
    background: #faf9ff;
    border: 1px solid #ede9fe;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

.result-tile .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}

.result-tile .value {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #1e1b4b;
}

.confidence-bar-wrap {
    background: #ede9fe;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 0.6rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #7c3aed, #fbbf24);
}

.severity-badge {
    display: inline-block;
    padding: 0.28rem 0.9rem;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.sev-mild     { background: #d1fae5; color: #065f46; }
.sev-moderate { background: #fef3c7; color: #92400e; }
.sev-severe   { background: #fee2e2; color: #991b1b; }

/* ── ADVISORY BOX ── */
.advisory-box {
    background: linear-gradient(135deg, #faf9ff, #f5f3ff);
    border-left: 3px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    font-size: 0.91rem;
    line-height: 1.85;
    color: #2d2a5e;
}

/* ── DIVIDER ── */
.section-divider {
    border: none;
    border-top: 1px solid #ede9fe;
    margin: 1.5rem 0;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 2.5rem 1rem;
    border-top: 1px solid #ede9fe;
    margin-top: 2rem;
    background: white;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #fbbf24);
}

/* ── SELECTBOX & FILE UPLOADER ── */
.stSelectbox > div > div {
    border-radius: 8px !important;
    border-color: #c4b5fd !important;
    background: #faf9ff !important;
}

.stSelectbox > div > div:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

/* ── INPUTS GENERAL ── */
.stSelectbox label, .stFileUploader label {
    color: #4f46e5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border-radius: 10px;
    border: 2px dashed #c4b5fd !important;
    background: #faf9ff !important;
    padding: 0.5rem;
    transition: border-color 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7c3aed !important;
}

/* ── UPLOADED IMAGE ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 2px solid #ede9fe;
    box-shadow: 0 4px 20px rgba(124,58,237,0.1);
}

/* ── DASHBOARD CONTENT PADDING ── */
.dashboard-wrap {
    padding: 2rem 2.5rem;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .landing-title { font-size: 2.4rem; }
    .landing-sub   { font-size: 0.95rem; }
    .result-grid   { grid-template-columns: 1fr; }
    .topbar        { padding: 0.8rem 1.2rem; }
    .panel         { padding: 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# =============================
# 🔹 LANDING PAGE
# =============================
if "app_started" not in st.session_state:
    st.session_state.app_started = False

if not st.session_state.app_started:
    st.markdown("""
    <div class='landing-wrapper'>
        <div class='landing-badge'>🌱 AI-Powered Precision Agriculture</div>
        <div class='landing-title'>SmartCrop<br><span>AI Advisor</span></div>
        <div class='landing-sub'>
            Detect crop diseases in seconds using deep learning.<br>
            Get severity analysis and expert treatment recommendations.
        </div>
        <div class='landing-features'>
            <div class='feat-chip'>🔬 Disease Detection</div>
            <div class='feat-chip'>🗺️ Severity Mapping</div>
            <div class='feat-chip'>🧠 AI Advisory</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1.5, 1, 1.5])
    with col_m:
        if st.button("🚀 Get Started", use_container_width=True, key="get_started_btn"):
            st.session_state.app_started = True
            st.rerun()

    st.stop()

# =============================
# 🔹 TOP BAR
# =============================
st.markdown("""
<div class='topbar'>
    <div class='topbar-logo'>🌿 Smart<span>Crop</span></div>
    <div class='topbar-tag'>AI Precision Agriculture</div>
</div>
""", unsafe_allow_html=True)

# =============================
# 🔹 MAIN DASHBOARD
# =============================
st.markdown("<div class='dashboard-wrap'>", unsafe_allow_html=True)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Configuration</div>", unsafe_allow_html=True)

    selected_plant = st.selectbox(
        "🌱 Select Plant Type",
        plant_names,
        key="plant_selector"
    )

    # Fix for Android WebView APK — forces re-render after selection
    if st.session_state.get("_last_plant") != selected_plant:
        st.session_state["_last_plant"] = selected_plant
        st.rerun()

    uploaded_file = st.file_uploader(
        "📤 Upload Leaf Image",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 🔹 PROCESS IMAGE
# =============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Uploaded Image</div>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Action Buttons ──
    with col1:
        b1, b2 = st.columns(2, gap="small")
        with b1:
            detect_clicked = st.button("🔍 Detect Disease", use_container_width=True)
        with b2:
            segment_clicked = st.button("🎯 Show Disease Area", use_container_width=True)

    # =============================
    # 🔹 DETECT DISEASE
    # =============================
    if detect_clicked:
        with st.spinner("Analyzing plant health..."):
            plant, disease, confidence = predict(image, selected_plant)

        with col2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Detection Results</div>", unsafe_allow_html=True)

            conf_pct = confidence * 100
            st.markdown(f"""
            <div class='result-grid'>
                <div class='result-tile'>
                    <div class='label'>Plant</div>
                    <div class='value'>🌿 {plant}</div>
                </div>
                <div class='result-tile'>
                    <div class='label'>Disease</div>
                    <div class='value'>🦠 {disease}</div>
                </div>
            </div>
            <div class='result-tile' style='margin-bottom:1rem;'>
                <div class='label'>Confidence Score</div>
                <div class='value'>{conf_pct:.1f}%</div>
                <div class='confidence-bar-wrap'>
                    <div class='confidence-bar-fill' style='width:{conf_pct:.1f}%'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Farmer Advisory</div>", unsafe_allow_html=True)

            with st.spinner("Generating advisory..."):
                explanation = get_disease_info(plant, disease)

            st.markdown(f"<div class='advisory-box'>{explanation}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # =============================
    # 🔹 SEGMENT DISEASE AREA
    # =============================
    if segment_clicked:
        with st.spinner("Analyzing infected regions..."):
            segmented, severity_percent = segment_image(image)
            severity = (
                "Mild" if severity_percent < 10
                else "Moderate" if severity_percent < 30
                else "Severe"
            )
            sev_class = f"sev-{severity.lower()}"

        with col2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Disease Segmentation</div>", unsafe_allow_html=True)

            st.image(segmented, caption="Infected regions highlighted in red", use_container_width=True)

            st.markdown(f"""
            <div class='result-grid' style='margin-top:1rem;'>
                <div class='result-tile'>
                    <div class='label'>Severity Level</div>
                    <div class='value'>
                        <span class='severity-badge {sev_class}'>{severity}</span>
                    </div>
                </div>
                <div class='result-tile'>
                    <div class='label'>Infected Area</div>
                    <div class='value'>{severity_percent:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(int(severity_percent), 100))
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close dashboard-wrap

# =============================
# 🔹 FOOTER
# =============================
st.markdown("""
<div class='footer'>
    <div style='font-size:1.1rem; font-weight:800; font-family:Syne,sans-serif; color:#4f46e5; margin-bottom:0.4rem; letter-spacing:-0.5px;'>
        🌿 Smart<span style='color:#fbbf24'>Crop</span> AI
    </div>
    Final Year Project &nbsp;·&nbsp; AI for Precision Agriculture &nbsp;·&nbsp; Deep Learning Disease Detection
</div>
""", unsafe_allow_html=True)
