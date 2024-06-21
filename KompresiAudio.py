import numpy as np
import pywt
import wave
import struct
from io import BytesIO
import streamlit as st
from scipy.fftpack import dct, idct
from pydub import AudioSegment
import tempfile

# Fungsi untuk membaca file audio
def read_audio(file_bytes):
    with wave.open(BytesIO(file_bytes), 'rb') as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
        audio = struct.unpack_from("%dh" % params.nframes, frames)
        audio = np.array(audio, dtype=np.int16)
    return audio, params

# Fungsi untuk menulis file audio
def write_audio(audio, params, format='wav'):
    with tempfile.NamedTemporaryFile(delete=False) as temp_wav_file:
        with wave.open(temp_wav_file.name, 'wb') as wav_file:
            wav_file.setparams(params)
            frames = struct.pack("%dh" % len(audio), *audio)
            wav_file.writeframes(frames)

        # Load temporary WAV file and export to MP3 using pydub
        audio_segment = AudioSegment.from_wav(temp_wav_file.name)
        mp3_filename = temp_wav_file.name.replace('.wav', '.mp3')
        audio_segment.export(mp3_filename, format='mp3')

    # Read the MP3 file back as bytes
    with open(mp3_filename, 'rb') as mp3_file:
        mp3_bytes = mp3_file.read()

    return mp3_bytes

# Fungsi untuk kompresi audio menggunakan DWT
def dwt_compress(audio, wavelet='db1', level=1):
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745
    coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    return coeffs, threshold

# Fungsi untuk dekompresi audio menggunakan DWT
def dwt_decompress(coeffs, wavelet='db1'):
    audio_reconstructed = pywt.waverec(coeffs, wavelet)
    audio_reconstructed = np.clip(audio_reconstructed, -32768, 32767)
    audio_reconstructed = np.array(audio_reconstructed, dtype=np.int16)
    return audio_reconstructed

# Fungsi untuk kompresi audio menggunakan DCT
def dct_compress(audio, quality=50):
    audio = audio - np.mean(audio)
    dct_audio = dct(audio, norm='ortho')
    threshold = np.percentile(np.abs(dct_audio), quality)
    dct_audio[np.abs(dct_audio) < threshold] = 0
    compressed_audio = idct(dct_audio, norm='ortho')
    compressed_audio = np.clip(compressed_audio, -32768, 32767)
    return np.array(compressed_audio, dtype=np.int16), threshold
