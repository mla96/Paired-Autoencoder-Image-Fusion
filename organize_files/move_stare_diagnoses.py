import os
import shutil


path = '../../../../Prince/Transfer_learning_datasets/'
move_from = 'train'
move_to = 'train_AMD'
txt_file = 'STARE/AMD_diag_subjs_list.txt'

f = open(os.path.join(path, txt_file), 'r')
lines = f.readlines()
f.close()

files = os.listdir(os.path.join(path, move_from))

for file in files:
    if 'STARE' in file:
        for line in lines:
            line = line.strip()
            if line in file:
                print(file)
                shutil.move(os.path.join(path, move_from, file),
                            os.path.join(path, move_to, file))
