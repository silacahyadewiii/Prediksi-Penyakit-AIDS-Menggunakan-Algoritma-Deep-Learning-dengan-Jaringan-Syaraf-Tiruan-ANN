import streamlit as st
import numpy as np
import tensorflow as tf

# Judul aplikasi
st.title("Prediksi AIDS Virus Infection")
st.write("Aplikasi ini memprediksi infeksi virus AIDS berdasarkan data numerik yang diinputkan.")

# Load model
try:
    model = tf.keras.models.load_model('best_ann_model.h5')
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Input manual untuk setiap kolom (tanpa 'infected')
st.write("## Prediksi dengan Input Manual")

# Input untuk masing-masing fitur (22 kolom)
time = st.number_input("Masukkan nilai untuk time", value=0.0, min_value=0.0, step=1.0)
trt = st.number_input("Masukkan nilai untuk trt", value=0.0, min_value=0.0, max_value=3.0, step=1.0)
age = st.number_input("Masukkan nilai untuk age", value=0.0, min_value=0.0, step=1.0)
wtkg = st.number_input("Masukkan nilai untuk wtkg", value=0.0, min_value=0.0, step=0.001)
hemo = st.number_input("Masukkan nilai untuk hemo", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
homo = st.number_input("Masukkan nilai untuk homo", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
drugs = st.number_input("Masukkan nilai untuk drugs", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
karnof = st.number_input("Masukkan nilai untuk karnof", value=0.0, min_value=0.0, max_value=100.0, step=1.0)
oprior = st.number_input("Masukkan nilai untuk oprior", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
z30 = st.number_input("Masukkan nilai untuk z30", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
preanti = st.number_input("Masukkan nilai untuk preanti", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
race = st.number_input("Masukkan nilai untuk race", value=0.0, min_value=0.0, step=1.0)
gender = st.number_input("Masukkan nilai untuk gender", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
str2 = st.number_input("Masukkan nilai untuk str2", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
strat = st.number_input("Masukkan nilai untuk strat", value=0.0, min_value=0.0, max_value=3.0, step=1.0)
symptom = st.number_input("Masukkan nilai untuk symptom", value=0.0, min_value=0.0, max_value=3.0, step=1.0)
treat = st.number_input("Masukkan nilai untuk treat", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
offtrt = st.number_input("Masukkan nilai untuk offtrt", value=0.0, min_value=0.0, max_value=1.0, step=1.0)
cd40 = st.number_input("Masukkan nilai untuk cd40", value=0.0, min_value=0.0, step=1.0)
cd420 = st.number_input("Masukkan nilai untuk cd420", value=0.0, min_value=0.0, step=1.0)
cd80 = st.number_input("Masukkan nilai untuk cd80", value=0.0, min_value=0.0, step=1.0)
cd820 = st.number_input("Masukkan nilai untuk cd820", value=0.0, min_value=0.0, step=1.0)

# Button untuk melakukan prediksi
if st.button("Prediksi"):
    try:
        # Mengumpulkan data input (tanpa kolom 'infected')
        input_data = np.array([
            time, trt, age, wtkg, hemo, homo, drugs, karnof, oprior, z30, preanti, 
            race, gender, str2, strat, symptom, treat, offtrt, cd40, cd420, cd80, cd820
        ]).reshape(1, -1)  # Hapus 'infected'

        # Debugging: Print data input ke aplikasi
        st.write("Data input:", input_data)

        # Melakukan prediksi
        prediction = model.predict(input_data)
        st.write("Hasil prediksi mentah:", prediction)

        # Ambil kelas prediksi
        pred_class = np.argmax(prediction, axis=1)[0]
        pred_label = "Terinfeksi" if pred_class == 1 else "Tidak Terinfeksi"
        st.success(f"Hasil prediksi: {pred_label}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
