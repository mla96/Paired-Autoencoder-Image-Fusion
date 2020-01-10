import os
import shutil


os.getcwd()
os.chdir("../../../../")

print(os.getcwd())

path = 'OakHome/FLIO/Data/AMD_subj_list'

f = open(path, 'r')
lines = f.readlines()
f.close()

for line in lines:
    line = line.strip()
    files = os.listdir(os.path.join('Documents/FLIMX-master/studyData/AMD/', line.replace('_',  '')))
    for file in files:
        shutil.copy(os.path.join('Documents/FLIMX-master/studyData/AMD', line.replace('_',  ''), file), os.path.join('OakHome/FLIO/Data', line))
