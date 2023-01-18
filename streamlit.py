import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
import wavio
from pydub import AudioSegment
from pathlib import Path
import ffmpeg
from librosa.core import resample, to_mono

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask, y_mean


def downsample_mono(path, sr):
    print(path)
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate

    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc

    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)

    return sr, wav

def upload_and_save_wavfiles(uploaded_file, save_dir):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('wav'):
            audio = AudioSegment.from_wav(uploaded_file)
            file_type = 'wav'

        save_path = Path(save_dir) / uploaded_file.name
        audio.export(save_path, format=file_type)

        return save_path


def classify_chord(wavpath):
    model = load_model('models/lstm_no_octave.h5',
                    custom_objects={'STFT':STFT,
                                    'Magnitude':Magnitude,
                                    'ApplyFilterbank':ApplyFilterbank,
                                    'MagnitudeToDecibel':MagnitudeToDecibel})

    classes = ['Af_aug', 'Af_dim', 'Af_major', 'Af_minor', 
               'An_aug', 'An_dim', 'An_major', 'An_minor', 
               'Bf_aug', 'Bf_dim', 'Bf_major', 'Bf_minor', 
               'Bn_aug', 'Bn_dim', 'Bn_major', 'Bn_minor', 
               'Cn_aug', 'Cn_dim', 'Cn_major', 'Cn_minor', 
               'Df_aug', 'Df_dim', 'Df_major', 'Df_minor', 
               'Dn_aug', 'Dn_dim', 'Dn_major', 'Dn_minor', 
               'Ef_aug', 'Ef_dim', 'Ef_major', 'Ef_minor', 
               'En_aug', 'En_dim', 'En_major', 'En_minor', 
               'Fn_aug', 'Fn_dim', 'Fn_major', 'Fn_minor', 
               'Gf_aug', 'Gf_dim', 'Gf_major', 'Gf_minor', 
               'Gn_aug', 'Gn_dim', 'Gn_major', 'Gn_minor']

    if wavpath is not None:
        rate, wav = downsample_mono(str(wavpath), 16000)
        mask, env = envelope(wav, rate, threshold=20)
        clean_wav = wav[mask]
        step = int(16000*1.0)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)

    st.text('Predicted Chord: {}'.format(classes[y_pred]))

# Start of Streamlit app
st.write("Here's our first attempt at musical chord recognition:")

# Get uploaded audio chord file
uploaded_file = st.file_uploader("Upload audio file (.wav format, piano chord only)", type=['wav'])

# Display audio player
audio_bytes = uploaded_file.read()
st.audio(audio_bytes, format='audio/wav')

# Temporary download uploaded file
uploaded_path = upload_and_save_wavfiles(uploaded_file, 'temp_uploaded')

# Classify the audio chord
classify_chord(uploaded_path)




      