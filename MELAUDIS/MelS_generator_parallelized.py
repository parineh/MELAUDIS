import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time

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

def process_one(task, FRAME_SIZE = 512, HOP_SIZE = 64):
    dir_name, file_path, save_filename = task

    # Load and normalize
    audio_unormed, sample_rate = librosa.load(file_path, sr=None)
    audio = normalize(audio_unormed)

    # Mel + log
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=64
    )
    log_mels = librosa.power_to_db(mel, ref=np.max)

    # Check output dir exists
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)

    # Save image
    save_spectrogram(log_mels, sample_rate, HOP_SIZE, save_filename)

    return os.path.basename(file_path)  # for logging

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

    FRAME_SIZE = 512
    HOP_SIZE = 64

    # 1) Build tasks
    tasks = []
    for dir_name in folders.values():
        curr_folder = os.path.join(dataset_organized, dir_name)

        for filename in os.listdir(curr_folder):
            if filename == "desktop.ini":
                continue

            file_path = os.path.join(curr_folder, filename)
            if not os.path.isfile(file_path):
                continue

            save_filename = os.path.join(
                dataset_with_MelS_organized,
                dir_name,
                f"{filename}__{FRAME_SIZE}-{HOP_SIZE}.png"
            )
            tasks.append((dir_name, file_path, save_filename))

    print("Total files to process:", len(tasks))

    # 2) Parallel execution
    i = 0
    t_start = time.perf_counter()

    max_workers = os.cpu_count()  
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futures):
            name = fut.result()  # se crasha qui vedi l'eccezione
            i += 1
            print(f"Processed file {i}: {name}")

    t_end = time.perf_counter()
    print(f"Total files processed: {i} in {t_end - t_start:.2f} s")
