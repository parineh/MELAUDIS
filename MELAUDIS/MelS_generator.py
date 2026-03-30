import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np

# Function to save the spectrogram as an image
def save_spectrogram(Y, sr, hop_length, save_path, y_axis="mel"):
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, y_axis=y_axis)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to normalize an audio signal to [-1, 1]
def normalize(audio):
    return 2 * ((audio - np.min(audio)) / np.ptp(audio)) - 1


if __name__ == "__main__":

    # Paths for audio files (input)
    dataset_organized = r"./dataset_organized"

    # Creating the directories for saving spectrograms (output)
    dataset_with_MelS_organized = r"./dataset_with_MelS_organized"
    
    folders = {"Train Vehicle":"Train/Veh", "Train No Vehicle":"Train/NoVeh", "Test Vehicle":"Test/Veh", "Test No Vehicle":"Test/NoVeh"}

    if not os.path.exists(dataset_with_MelS_organized):
        for dir in folders.values():
            os.makedirs(os.path.join(dataset_with_MelS_organized, dir))
        print("dataset_with_MelS_organized directory created succesfully")
    else: print("dataset_with_MelS_organized directory already exist")


    # Parameters
    FRAME_SIZE = 512
    HOP_SIZE = 64

    # Process each file in the Car directory and save the spectrograms
    i = 0
    for dir in folders.values():
        curr_folder = os.path.join(dataset_organized, dir)
        for filename in os.listdir(curr_folder):
            if filename != 'desktop.ini':  # Skip system files
                file_path = os.path.join(curr_folder, filename)
                
                if os.path.isfile(file_path):
                    # Load and normalize the audio
                    audio_car_unormed, sample_rate = librosa.load(file_path, sr=None)
                    audio_car = normalize(audio_car_unormed)

                    # Create mel-spectrogram and convert to log scale
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio_car, sr=sample_rate, 
                                                                    n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=64)
                    log_mels = librosa.power_to_db(mel_spectrogram, ref=np.max)

                    # Set the directory and filename for saving
                    save_filename = f"{os.path.join(dataset_with_MelS_organized, dir)}/{filename}__{FRAME_SIZE}-{HOP_SIZE}.png"
                    
                    # Save the Log-Amplitude Mel-Spectrogram
                    save_spectrogram(log_mels, sample_rate, HOP_SIZE, save_filename)
                    i += 1
                    print(f"Processed file {i}: {filename}")
    print(f"Total files processed: {i}")
