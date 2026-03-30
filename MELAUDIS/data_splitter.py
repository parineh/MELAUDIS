import os
import shutil
import time


def move_all_to_tmp(directory, tmp):
    n_moved = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"): 
                shutil.move(os.path.join(root, file), tmp)
                n_moved += 1
    return n_moved
                    
def count_tot_wav(directory):
    n = 0
    for _, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                n += 1
    return n

def split_data(train_size, dir, folders, Vehicle=True):
    for _ in range(train_size):
        n_moved = 0
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(".wav"): 
                    if n_moved < train_size:
                            shutil.move(os.path.join(root, file), os.path.join(dataset_organized, folders["Train Vehicle"] if Vehicle else folders["Train No Vehicle"]))
                    else: shutil.move(os.path.join(root, file), os.path.join(dataset_organized, folders["Test Vehicle"] if Vehicle else folders["Test No Vehicle"]))
                    n_moved += 1
        return n_moved

if __name__ == "__main__":
    directory_path_Veh = r"./dataset/MELAUDIS_Vehicles/Final_Veh"
    directory_path_NoVeh = r"./dataset./MELAUDIS_ BG/_BG_Final"
    tmp_Veh = r"./tmp"

    if not os.path.exists(tmp_Veh):
        os.makedirs(tmp_Veh)
        print("tmp directory created succesfully")
    else: print("tmp directory already exist")

    n_wav_Veh = count_tot_wav(directory_path_Veh)
    print(f"\nTotal wav found: {n_wav_Veh}")
    print("Starting moving:")
    t_start = time.perf_counter()
    n_moved = move_all_to_tmp(directory_path_Veh, tmp_Veh)
    t_end = time.perf_counter()
    print(f"Moved {n_moved}/{n_wav_Veh} in {t_end - t_start:.2f} s")


    dataset_organized = r"./dataset_organized"
    folders = {"Train Vehicle":"Train/Veh", "Train No Vehicle":"Train/NoVeh", "Test Vehicle":"Test/Veh", "Test No Vehicle":"Test/NoVeh"}

    if not os.path.exists(dataset_organized):
        for dir in folders.values():
            os.makedirs(os.path.join(dataset_organized, dir))
        print("dataset_organized directory created succesfully")
    else: print("dataset_organized directory already exist")

    n_wav_Veh = count_tot_wav(tmp_Veh)
    train_size_fraction_Veh = 0.8
    train_size_Veh = int(n_wav_Veh * 0.8)
    split_data(train_size_Veh, tmp_Veh, folders, Vehicle=True)

    n_wav_NoVeh = count_tot_wav(directory_path_NoVeh)
    train_size_fraction_NoVeh = 0.8
    train_size_NoVeh = int(n_wav_NoVeh * 0.8)
    split_data(train_size_NoVeh, directory_path_NoVeh, folders, Vehicle=False)