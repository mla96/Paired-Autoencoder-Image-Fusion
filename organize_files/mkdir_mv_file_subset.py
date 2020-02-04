import os
import shutil


os.getcwd()
os.chdir("../../Data")

# print(os.getcwd())

path = 'AMD_subj_list'
creation_path = 'registration_fixed_jpg_moving_sdt'

f = open(path, 'r')
lines = f.readlines()
f.close()

for line in lines:
    line = line.strip()
    current_dir = os.path.join(os.getcwd(), line)
    creation_dir = os.path.join(current_dir, creation_path)
    if not os.path.exists(creation_dir):
        os.makedirs(creation_dir)

    for file in os.listdir(current_dir):
        if file.startswith('checker') or file.startswith('registered'):
            file_to_move = os.path.join(current_dir, file)
            shutil.move(file_to_move, creation_dir)
