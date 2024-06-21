from PIL import Image
import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import os
import numpy as np
import cv2
import pywt
import tempfile
from scipy.fftpack import dct, idct

# Fungsi untuk mengompres frame menggunakan DCT
def compress_frame_dct(frame):
    frame = np.float32(frame) / 255.0
    frame_dct = dct(dct(frame.T, norm='ortho').T, norm='ortho')
    frame_dct[20:, 20:] = 0  # Buang frekuensi tinggi
    frame_idct = idct(idct(frame_dct.T, norm='ortho').T, norm='ortho')
    frame_compressed = np.uint8(np.clip(frame_idct * 255.0, 0, 255))
    return frame_compressed

# Fungsi untuk mengompres frame menggunakan DWT
def compress_frame_dwt(frame):
    frame = np.float32(frame) / 255.0
    coeffs = pywt.dwt2(frame, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH.fill(0)
    cV.fill(0)
    cD.fill(0)
    frame_idwt = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    frame_compressed = np.uint8(np.clip(frame_idwt * 255.0, 0, 255))
    return frame_compressed

# Fungsi untuk melakukan kompresi video menggunakan DCT
def compress_video_dct(video_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
            temp_in.write(video_bytes)
            temp_in.close()
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            cap = cv2.VideoCapture(temp_in.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_compressed = compress_frame_dct(frame)
                out.write(frame_compressed)

            cap.release()
            out.release()

            with open(temp_out, 'rb') as f:
                compressed_video = f.read()

            os.remove(temp_in.name)
            os.remove(temp_out)

            return compressed_video
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Fungsi untuk melakukan kompresi video menggunakan DWT
def compress_video_dwt(video_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
            temp_in.write(video_bytes)
            temp_in.close()
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            cap = cv2.VideoCapture(temp_in.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_compressed = compress_frame_dwt(frame)
                out.write(frame_compressed)

            cap.release()
            out.release()

            with open(temp_out, 'rb') as f:
                compressed_video = f.read()

            os.remove(temp_in.name)
            os.remove(temp_out)

            return compressed_video
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Fungsi untuk menampilkan tombol unduh
def download_button(image_bytes, file_name):
    st.download_button(
        label="Unduh Gambar Kompresi",
        data=image_bytes,
        file_name=file_name,
        mime="image/jpeg"
    )