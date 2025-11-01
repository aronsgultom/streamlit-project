import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import uuid
import json

# =========================================
# Load model dari folder project (Git LFS)
MODEL_PATH = "model_tomat.keras"
if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# Load label map
LABEL_MAP_PATH = "label_map.json"
if not os.path.exists(LABEL_MAP_PATH):
    st.error(f"File label map tidak ditemukan: {LABEL_MAP_PATH}")
    st.stop()

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

# Buat list label sesuai indeks
labels = [None] * len(label_map)
for label, index in label_map.items():
    labels[index] = label.replace('_', ' ').capitalize()

# Deskripsi penyakit
deskripsi_penyakit = {
    'Early blight': "Bercak gelap konsentris pada daun. Disebabkan jamur Alternaria solani.",
    'Late blight': "Bercak gelap dan basah, daun menguning dan membusuk.",
    'Leaf mold': "Bercak kuning di atas daun, jamur abu-abu di bawah daun.",
    'Healthy': "Daun dalam kondisi sehat."
}

# =========================================
# Streamlit UI
st.title("🍅 Klasifikasi Penyakit Daun Tomat")

uploaded_file = st.file_uploader("Pilih gambar daun tomat (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file:
    # Simpan file sementara
    upload_dir = "temp_upload"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}.png")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar
    st.image(file_path, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing gambar
    img = image.load_img(file_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)[0]
    top_index = np.argmax(pred)
    result = labels[top_index]

    # Probabilitas setiap kelas
    prediction_list = [(labels[i], f"{p*100:.2f}%") for i, p in enumerate(pred)]
    description = deskripsi_penyakit.get(result, "Deskripsi tidak tersedia.")

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"**{result}**")
    st.write(description)

    st.subheader("Probabilitas setiap kelas:")
    for label, prob in prediction_list:
        st.write(f"{label}: {prob}")
