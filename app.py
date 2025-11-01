import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import uuid
import json
import gdown

# =========================================
# Load model
MODEL_PATH = "model_tomat.keras"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=YOUR_DRIVE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# Load label map
with open("label_map.json") as f:
    label_map = json.load(f)
labels = [None]*len(label_map)
for label, index in label_map.items():
    labels[index] = label.replace('_',' ').capitalize()

# Deskripsi penyakit
deskripsi_penyakit = {
    'Early blight': "...",
    'Late blight': "...",
    'Leaf mold': "...",
    'Healthy': "..."
}

# =========================================
# Streamlit UI
st.title("🍅 Klasifikasi Penyakit Daun Tomat")
uploaded_file = st.file_uploader("Pilih gambar daun tomat", type=["jpg","jpeg","png"])

if uploaded_file:
    file_path = os.path.join("temp_upload", f"{uuid.uuid4().hex}.png")
    os.makedirs("temp_upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing dan prediksi
    img = image.load_img(file_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    top_index = np.argmax(pred)
    result = labels[top_index]

    prediction_list = [(labels[i], f"{p*100:.2f}%") for i, p in enumerate(pred)]
    description = deskripsi_penyakit.get(result, "Deskripsi tidak tersedia.")

    st.subheader("Hasil Prediksi:")
    st.write(f"**{result}**")
    st.write(description)

    st.subheader("Probabilitas setiap kelas:")
    for label, prob in prediction_list:
        st.write(f"{label}: {prob}")
