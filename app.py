from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import librosa.display
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
from scipy.io.wavfile import read
import numpy as np
data_dir = "input/"
os.listdir(data_dir)
import tensorflow as tf
from spela.spectrogram import Spectrogram 
from spela.melspectrogram import Melspectrogram

def create_model(speech_feature):
    model = tf.keras.Sequential()
    if speech_feature == "spectrogram":
        model.add(Spectrogram(input_shape=(1, 16000)))
    elif speech_feature == "melspectrogram":
         model.add(Melspectrogram(input_shape=(1, 16000), name='melgram'))

    model.add(tf.keras.layers.Conv2D(265, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4)
            , loss = "categorical_crossentropy"
            , metrics = ["accuracy"])
    return model

def load_wav(wav_path, speaker):
    wav_path = data_dir + speaker + "/" + wav_path
    
    wav_data, _ = tf.audio.decode_wav(tf.io.read_file(wav_path), desired_channels=1)
    wav_data = tf.reshape(wav_data, [1, -1])
    return wav_data

def add_awgn(signal, snr_dB):
    # Calculate noise power
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10.0))
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    # Add noise to the signal
    noisy_signal = signal + noise
    return noisy_signal

def generate_training_data(speaker_paths, speaker, label, snr_dB=15):
    wavs, labels = [], []
    for i in tqdm(speaker_paths):
        wav = load_wav(i, speaker)
        # Add AWGN to the waveform
        noisy_wav = add_awgn(wav, snr_dB)
        wavs.append(noisy_wav)
        labels.append(label)
    return wavs, labels

def get_wav_paths(speaker):
    speaker_path = data_dir + speaker
    all_paths = [item for item in os.listdir(speaker_path)]
    return all_paths

def adpred(filename):
    nelson_madela= [item for item in os.listdir(data_dir + "Benjamin_Netanyau")]
    nelson_madela[:10]

    benjamin_netanyau_paths = get_wav_paths("Benjamin_Netanyau")
    jens_stoltenberg_paths = get_wav_paths( 'Jens_Stoltenberg')
    julia_gillard_paths = get_wav_paths("Julia_Gillard")


    benjamin_netanyau_wavs, benjamin_netanyau_labels = generate_training_data(benjamin_netanyau_paths, "Benjamin_Netanyau", 2) 
    jens_stoltenberg_wavs, jens_stoltenberg_labels = generate_training_data(jens_stoltenberg_paths, "Jens_Stoltenberg", 3) 
    julia_gillard_wavs, julia_gillard_labels = generate_training_data(julia_gillard_paths, "Julia_Gillard", 4)

# remove the extra wav for Julia Gillard
    julia_gillard_labels = julia_gillard_labels[1:]
    julia_gillard_wavs = julia_gillard_wavs[1:]

    all_wavs = benjamin_netanyau_wavs + jens_stoltenberg_wavs + julia_gillard_wavs
    all_labels = benjamin_netanyau_labels + jens_stoltenberg_labels + julia_gillard_labels
    train_wavs, test_wavs, train_labels, test_labels = train_test_split(all_wavs, all_labels, test_size=0.2)
    train_x, train_y = np.array(train_wavs), np.array(train_labels)
    test_x, test_y = np.array(test_wavs), np.array(test_labels)

    test_x.shape
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)



    model = create_model("spectrogram")
    model.summary()
    history=model.fit(x=train_x, y=train_y, epochs=20, validation_data=(test_x, test_y))
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print(f"loss: {test_loss:0.3}\t accuracy: {test_accuracy:0.1%}")
    y_predication = model.predict(test_x)



    file_path = filename

    wav_data, _ = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1, desired_samples=16000)
    reshaped_array = tf.reshape(wav_data, [1, 1, 16000])
    reshaped_array=add_awgn(reshaped_array,15)
    #audio_arrays.append(reshaped_array)
    audio_data_array = np.concatenate(reshaped_array, axis=0)

    print("Shape of the resulting array:", audio_data_array.shape)

    class_names = {2: "Jens_Stoltenberg", 1: "Benjamin_Netanyau",3:'Julia_Gillard',4:'Unkown'}
    y_pred=[]
    y_predication = model.predict(test_x)
    predicted_label = int(np.argmax(y_predication))
    print(predicted_label)
    if(predicted_label >3 and predicted_label==0):
        predicted_label=4
        
    return class_names[predicted_label]
    
app = Flask(__name__, template_folder='templates')

@app.route('/')
def Home():
    return render_template('index.html')



@app.route('/predicting', methods=['GET', 'POST'])
def predicting():
    if request.method == 'POST':
        file = request.files['file'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('./static/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        predictions=adpred(file_path)
    
    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run()
