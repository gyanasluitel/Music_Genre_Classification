import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import librosa
import librosa.display

import keras
from keras import layers, Sequential
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment

import platform

# Defining Constants
n_mfcc = 13
n_fft = 2048
num_segments = 10
hop_length = 512
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

expected_num_mfcc_vectors_per_segment = math.ceil(
    num_samples_per_segment / hop_length)


def generating_headers(n_mfcc=13):
    headers = []
    for i in range(1, n_mfcc+1):
        headers.append(f'mfcc_{i}')
    return headers


def predict_song(song_path):
    signal, sample_rate = librosa.load(song_path, sr=SAMPLE_RATE)
    song_length = int(librosa.get_duration(filename=song_path))

    if song_length > 30:
        # print(f"Song is greater than 30 seconds")
        discarded_song_length = (song_length % 30)
        song_length -= discarded_song_length

        SAMPLES_PER_SONG = SAMPLE_RATE * song_length
        parts = int(song_length/30)
        SAMPLES_PER_SEGMENT_30 = int(SAMPLES_PER_SONG/(parts))
        flag = 1

    elif song_length == 30:
        parts = 1
        flag = 0
    else:
        print("Too short, enter a song of length minimum 30 seconds")
        flag = 2

    single_file_df = pd.DataFrame()
    headers = generating_headers()
    for i in range(0, parts):
        if flag == 1:
            # print(f'Song snipped: {i+1}')
            # calculate start and finish sample for current part
            start30 = SAMPLES_PER_SEGMENT_30 * i
            finish30 = start30 + SAMPLES_PER_SEGMENT_30
            y = signal[start30:finish30]
        elif flag == 0:
            print('Song is 30 seconds long so no need for slicing.')
            y = signal

        for segment in range(num_segments):
            # calculate start and finish sample for current segment
            start_sample = num_samples_per_segment * segment
            finish_sample = start_sample + num_samples_per_segment

            # extract mfcc
            mfcc = librosa.feature.mfcc(
                y[start_sample:finish_sample], sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            # print(f'Shape of mfcc: {mfcc.shape}')
            # mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1],1)
            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                segment_df = pd.DataFrame(data=mfcc,
                                          columns=headers,
                                          index=range(len(mfcc)))
                single_file_df = pd.concat([single_file_df, segment_df],
                                           axis=0,
                                           sort=False,
                                           ignore_index=True)

    if flag == 1:
        single_file_df = single_file_df.iloc[520: len(single_file_df) - 520]

    song_values = []
    for i in range(0, single_file_df.shape[0], 130):
        # print(i, i+130)
        # i+1
        # if i>=song_df2.shape[0]+130:
        #     break
        song_values.append(single_file_df.iloc[i:i+130, :])
        i += 1
        if i >= single_file_df.shape[0]+130:
            break

    predictions = []
    for x in range(len(song_values)):
        X_to_predict = np.expand_dims(song_values[x], 0)
        X_to_predict = np.expand_dims(X_to_predict, 3)
        # print(X_to_predict.shape)
        test = np.array(X_to_predict, dtype=np.float32)
        try:
            result = cnn_model.predict(test)
            predictions.append(result)
        except Exception as e:
            print(f'{e}')

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
              'metal', 'pop', 'reggae', 'rock']
    all_prediction_df = pd.DataFrame(columns=genres)
    for prediction in predictions:
        predict_row = pd.DataFrame(data=prediction, columns=genres)
        all_prediction_df = pd.concat(
            [all_prediction_df, predict_row], ignore_index=True)
    # print("")
    # print(f"Predicted genre of the song: ",
    #       all_prediction_df.apply(np.mean, axis=0).idxmax())
    predicted_mean = all_prediction_df.apply(np.mean, axis=0)
    predicted_label = predicted_mean.idxmax()
    return predicted_mean, predicted_label


def generating_mfcc_plot(song_path):
    signal, sample_rate = librosa.load(song_path, sr=SAMPLE_RATE)

    MFCCs = librosa.feature.mfcc(signal,
                                 sample_rate,
                                 n_fft=n_fft,
                                 hop_length=hop_length,
                                 n_mfcc=n_mfcc)

    fig, ax = plt.subplots(figsize=(8, 6))

    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.title("MFCCs")
    plt.colorbar()
    st.pyplot(fig)


@st.cache(allow_output_mutation=True)
def load_model():
    dirname = os.path.dirname(__file__)
    system_name = platform.system()
    if system_name == 'Linux':
        os.chdir(dirname + '/..')
        main_path = os.getcwd()
        # print(main_path)
        model_path = os.path.join(main_path, 'models/CNN_model_mfcc13_G.h5')
    elif system_name == 'Windows':
        os.chdir(dirname + '\..')
        main_path = os.getcwd()
        # print(main_path)
        model_path = os.path.join(main_path, 'models\CNN_model_mfcc13_G.h5')

    cnn_model = keras.models.load_model(model_path)
    cnn_model.summary()
    os.chdir(dirname)
    return cnn_model


if __name__ == '__main__':
    cnn_model = load_model()
    header_html = '''
        <h1 style='text-align:center; font-size:56px'>Music Genre Classifier</h1>
        <hr/>
    '''
    st.markdown(header_html, unsafe_allow_html=True)
    file = st.sidebar.file_uploader("Please Upload a wav Audio File",
                                    type=["wav"])

    if file is not None:
        datapath = os.getcwd()
        file_var = AudioSegment.from_wav(file)
        print('Exporting')
        dirname = os.path.dirname(__file__)
        song_path = os.path.join(dirname, file.name)
        print(song_path)
        file_var.export(song_path, format='wav')
        print('Done Exporting')

        st.sidebar.write("**Play the Song!**")
        st.sidebar.audio(file, "audio/wav")

    if st.sidebar.button('Predict the song!'):
        if file is None:
            st.error("Please upload a file")
        else:
            with st.spinner("Predicting..."):
                predicted_mean, predicted_label = predict_song(song_path)
                plt.style.use(['seaborn'])
                fig, ax = plt.subplots(figsize=(8, 6))

                ax.bar(x=predicted_mean.index, height=predicted_mean)

                st.write("Predicted Label: {}".format(predicted_label))

                plt.title('Probability Distribution of Predicted Genres')
                plt.xticks(rotation=90)
                plt.xlabel('Genres')
                plt.ylabel('Probability')

                st.pyplot(fig)

                st.write('The MFCC graph of the song:')
                generating_mfcc_plot(song_path)
                st.markdown('<hr/>', unsafe_allow_html=True)
