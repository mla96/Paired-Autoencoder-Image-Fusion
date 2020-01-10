import os
import re
import shutil


os.getcwd()
os.chdir("../../")

path = 'Data/AMD_subj_list'

f = open(path, 'r')
for line in f.readlines():
    line = line.strip()
    if not os.path.exists(os.path.join('Data', line)):
        os.mkdir(os.path.join('Data', line))
    files = [file for file in os.listdir('./Data') if re.match(re.compile(line + '.+'), file)]
    for file in files:
        shutil.move(os.path.join('Data', file), os.path.join('Data', line))
    print(line)
f.close()