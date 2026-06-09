import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image
import pytesseract

# ── Tesseract path ────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'D:\handwritten_character_recognition\tesseract.exe'

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODEL_PATH = "emnist_cnn_model.h5"

# ── Build & train model ───────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)

    st.info("Training CNN on EMNIST Letters — ~3 minutes, only once...")
    import tensorflow_datasets as tfds

    def preprocess(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.image.flip_left_right(img)
        label = sample['label'] - 1
        return img, label

    ds_train = tfds.load('emnist/letters', split='train', as_supervised=False)
    ds_test  = tfds.load('emnist/letters', split='test',  as_supervised=False)
    ds_train = ds_train.map(preprocess).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(ds_train, epochs=10, validation_data=ds_test)
    model.save(MODEL_PATH)
    st.success("✅ Model trained and saved!")
    return model

# ── Preprocess single character image ────────────────────────────────────────
def preprocess_image(gray_img):
    resized = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)

# ── Show prediction results ───────────────────────────────────────────────────
def show_prediction(model, input_arr):
    preds = model.predict(input_arr, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    top5_labels = [LABELS[i] for i in top5_idx]
    top5_scores = [preds[i] * 100 for i in top5_idx]
    predicted = LABELS[np.argmax(preds)]
    confidence = preds.max() * 100

    st.markdown(f"## Predicted: `{predicted}`")
    st.metric("Confidence", f"{confidence:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#2ecc71' if l == predicted else '#3498db' for l in top5_labels]
    ax.barh(top5_labels[::-1], top5_scores[::-1], color=colors[::-1])
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 5 Predictions")
    ax.set_xlim(0, 100)
    for i, (l, s) in enumerate(zip(top5_labels[::-1], top5_scores[::-1])):
        ax.text(s + 0.5, i, f"{s:.1f}%", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── OCR for full handwritten image ───────────────────────────────────────────
def extract_text_ocr(pil_image):
    img = np.array(pil_image.convert('L'))

    # Upscale for better OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise
    img = cv2.GaussianBlur(img, (1, 1), 0)

    # Binarize
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    processed = Image.fromarray(img)

    # PSM 6 = assume uniform block of text
    text = pytesseract.image_to_string(processed, config='--psm 6')
    return text.strip()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✍️ Handwritten Character Recognition")
st.markdown("**Tab 1:** Draw a single letter (CNN) &nbsp;|&nbsp; **Tab 2:** Upload handwritten image → extract full text (OCR)")
st.divider()

with st.spinner("Loading CNN model..."):
    model = get_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🖊️ Draw a Letter (CNN)", "📁 Upload Image → Extract Text (OCR)"])

# ════════════════════════════════════════════════════════
# TAB 1 — Draw single letter with CNN
# ════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖊️ Draw a single letter A–Z")
        st.caption("Use thick strokes, centered in the box")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        st.button("🗑️ Clear Canvas", use_container_width=True)

    with col2:
        st.subheader("🔍 CNN Prediction")
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            if img[:, :, :3].sum() > 1000:
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                input_arr = preprocess_image(gray)
                show_prediction(model, input_arr)
            else:
                st.info("Draw a letter on the canvas to see prediction.")
        else:
            st.info("Draw a letter on the canvas to see prediction.")

# ════════════════════════════════════════════════════════
# TAB 2 — Upload full handwritten image → OCR
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("📁 Upload a handwritten image")
    st.caption("Works best with clearly written text on white/light background. Supports printed and handwritten notes.")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded:
        pil_img = Image.open(uploaded)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        with col_b:
            if st.button("🔍 Extract Text", type="primary", use_container_width=True):
                with st.spinner("Running OCR..."):
                    result_text = extract_text_ocr(pil_img)

                if result_text:
                    st.subheader("📝 Extracted Text")
                    st.text_area("Result", result_text, height=350)
                    st.download_button(
                        label="⬇️ Download as .txt",
                        data=result_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No text detected. Try a clearer image with better contrast.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Srimukh Kandlikar · CNN + Tesseract OCR · Amazon ML School Project")