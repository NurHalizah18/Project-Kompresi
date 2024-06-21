import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import cv2
import pywt
import tempfile
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from KompresiGambar import compress_image_dct, compress_image_dwt, download_button
from KompresiAudio import read_audio, write_audio, dwt_compress, dwt_decompress, dct_compress
from KompresiVideo import compress_frame_dct, compress_frame_dwt, compress_video_dct, compress_video_dwt, download_button

def main():
    st.set_page_config(page_title='Project Kompresi')

    with st.sidebar:
        st.title('Menu Kompresi')
        options = {
            'Beranda': '🏠 ',
            'Kompresi Gambar': '🖼️ ',
            'Kompresi Audio': '🔊 ',
            'Kompresi Video': '📹 '
        }

        selected_option = st.selectbox('Pilih Menu', list(options.keys()), format_func=lambda x: f'{options[x]} {x}')

    if selected_option == 'Beranda':
        st.title('Beranda')
        st.markdown('## **Selamat Datang di Project Kompresi!**')
        st.write('Kompresi gambar, audio, dan video dengan DCT dan DWT memanfaatkan teknologi transformasi untuk mengurangi ukuran file tanpa mengorbankan kualitas. DCT membagi data menjadi komponen frekuensi untuk menghilangkan redundansi dalam gambar, audio, dan video, sementara DWT memisahkan informasi berdasarkan waktu dan frekuensi untuk kompresi yang lebih adaptif. Dengan pendekatan ini, media multimedia dapat disimpan dan dipertukarkan secara lebih efisien tanpa kehilangan detail esensial.')

    # KOMPRESI GAMBAR
    elif selected_option == 'Kompresi Gambar':
        st.title('Kompresi Gambar')
        st.write('Ini adalah halaman untuk kompresi gambar.')
        st.write("Muat gambar dan kompres dengan kualitas tertentu.")

        uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        algorithm_selected = st.selectbox("Pilih Algoritma: ", ["Discrete Cosine Transform (DCT)", "Discrete Wavelet Transform (DWT)"])

        quality = st.slider("Kualitas Kompresi (0-100)", 0, 100, 50)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            if algorithm_selected == "Discrete Cosine Transform (DCT)":
                compressed_image_bytes = compress_image_dct(image, quality)
            elif algorithm_selected == "Discrete Wavelet Transform (DWT)":
                compressed_image_bytes = compress_image_dwt(image, quality)

            st.image(compressed_image_bytes, caption=f"Gambar Kompresi dengan {algorithm_selected}")
            
            download_button(compressed_image_bytes, "compressed_image.jpg")
            
# KOMPRESI AUDIO
    elif selected_option == 'Kompresi Audio':
        st.title('Kompresi Audio')
        st.write("Unggah file audio (WAV) dan kompres dengan Discrete Wavelet Transform (DWT) atau Discrete Cosine Transform (DCT).")

        uploaded_file = st.file_uploader("Pilih file audio", type=["wav"], accept_multiple_files=False)

        if uploaded_file is not None:
            st.write('File yang diunggah:', uploaded_file.name)

            audio, params = read_audio(uploaded_file.getvalue())

            algorithm = st.selectbox("Pilih Algoritma Kompresi:", ["DWT", "DCT"])
            if algorithm == "DWT":
                wavelet = st.selectbox("Pilih Wavelet:", ['db1', 'haar', 'sym2', 'coif1'])
                level = st.slider("Pilih Level DWT:", 1, 5, 1)
            elif algorithm == "DCT":
                quality = st.slider("Pilih Kualitas Kompresi DCT (0-100):", 0, 100, 50)

            if st.button('Kompresi'):
                if algorithm == "DWT":
                    coeffs, threshold = dwt_compress(audio, wavelet=wavelet, level=level)
                    compressed_audio = dwt_decompress(coeffs, wavelet=wavelet)
                    file_name = "compressed_audio_dwt.wav"
                    st.write("DWT Threshold: ", threshold)
                elif algorithm == "DCT":
                    compressed_audio, threshold = dct_compress(audio, quality=quality)
                    file_name = "compressed_audio_dct.wav"
                    st.write("DCT Threshold: ", threshold)

                compressed_audio_bytes = write_audio(compressed_audio, params, format='mp3')
                st.audio(compressed_audio_bytes, format='audio/mp3', start_time=0)

                st.download_button(
                    label="Unduh Audio Kompresi",
                    data=compressed_audio_bytes,
                    file_name=file_name.replace('.wav', '.mp3'),
                    mime="audio/mp3"
                )
                st.success(f"Kompresi audio berhasil! File disimpan sebagai {file_name.replace('.wav', '.mp3')}")

                # Plot audio asli dan hasil kompresi untuk perbandingan
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(audio, label="Original Audio")
                ax[0].legend()
                ax[0].set_title("Original Audio")
                ax[1].plot(compressed_audio, label="Compressed Audio")
                ax[1].legend()
                ax[1].set_title("Compressed Audio")
                st.pyplot(fig)


    # KOMPRESI VIDEO
    elif selected_option == 'Kompresi Video':
        st.title('Kompresi Video')
        st.write("Pilih algoritma dan muat video untuk melakukan kompresi.")

        algorithm_selected = st.selectbox("Algoritma: ", ["Algoritma Discrete Cosine Transform (DCT)", "Algoritma Discrete Wavelet Transform (DWT)"])

        if algorithm_selected == "Algoritma Discrete Cosine Transform (DCT)":
            uploaded_file1 = st.file_uploader("Pilih file video", type=["mp4"], accept_multiple_files=False)

            if uploaded_file1 is not None:
                st.write('File yang diunggah:', uploaded_file1.name)

                if st.button('Kompresi Algoritma DCT'):
                    with st.spinner('Mengompresi video...'):
                        compressed_video = compress_video_dct(uploaded_file1.getvalue())
                    if compressed_video:
                        st.video(compressed_video, format='video/mp4', start_time=0)
                        download_button(compressed_video, "compressed_video_dct.mp4")
                        st.success("Kompresi video menggunakan Algoritma DCT berhasil!")
                    else:
                        st.error("Gagal mengompresi video menggunakan Algoritma DCT.")

        elif algorithm_selected == "Algoritma Discrete Wavelet Transform (DWT)":
            uploaded_file2 = st.file_uploader("Pilih file video", type=["mp4"], accept_multiple_files=False)

            if uploaded_file2 is not None:
                st.write('File yang diunggah:', uploaded_file2.name)

                if st.button('Kompresi Algoritma DWT'):
                    with st.spinner('Mengompresi video...'):
                        compressed_video = compress_video_dwt(uploaded_file2.getvalue())
                    if compressed_video:
                        st.video(compressed_video, format='video/mp4', start_time=0)
                        download_button(compressed_video, "compressed_video_dwt.mp4")
                        st.success("Kompresi video menggunakan Algoritma DWT berhasil!")
                    else:
                        st.error("Gagal mengompresi video menggunakan Algoritma DWT.")

if __name__ == '__main__':
    main()
