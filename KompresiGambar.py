from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import streamlit as st
import pywt


# Fungsi untuk melakukan kompresi gambar menggunakan DCT
def compress_image_dct(image, quality):
    img = image.copy()
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Fungsi untuk melakukan kompresi gambar menggunakan DWT
def compress_image_dwt(image, quality):
    img = np.array(image)

    # Gunakan wavelet 'haar' untuk DWT
    coeffs = pywt.dwt2(img, 'haar')

    # Ambil koefisien aproksimasi (cA) dan detail (cH, cV, cD)
    cA, (cH, cV, cD) = coeffs

    # Gunakan thresholding untuk kompresi dengan mengatur nilai threshold
    threshold = np.percentile(np.abs(cA), 100 - quality)
    cA_thresh = pywt.threshold(cA, threshold, mode='soft')

    # Rekonstruksi gambar dari koefisien yang telah diproses
    coeffs_thresh = (cA_thresh, (cH, cV, cD))
    reconstructed_image = pywt.idwt2(coeffs_thresh, 'haar')

    # Konversi kembali ke gambar dan pastikan dalam mode RGB sebelum disimpan
    compressed_image = Image.fromarray(np.uint8(reconstructed_image)).convert('RGB')

    buf = BytesIO()
    compressed_image.save(buf, format="JPEG")
    return buf.getvalue()


# Fungsi untuk menampilkan tombol unduh
def download_button(image_bytes, file_name):
    st.download_button(
        label="Unduh Gambar Kompresi",
        data=image_bytes,
        file_name=file_name,
        mime="image/jpeg"
    )
