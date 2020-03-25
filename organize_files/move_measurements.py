import os
import shutil


os.getcwd()
os.chdir("../../../../")

print(os.getcwd())

path = 'OakFundus/FLIO_Data/AMD_subj_list'
flimx_path = 'Documents/FLIMX_3/studyData/AMD_13/'

f = open(path, 'r')
lines = f.readlines()
f.close()

for line in lines:
    if line != 'AMD_01\n':
        line = line.strip()
        print(line)
        try:
            files = os.listdir(os.path.join(flimx_path, line.replace('_', '')))
            if not os.path.exists(os.path.join('OakFundus/FLIO_Data', line, 'FLIO_parameters')):
                os.mkdir(os.path.join('OakFundus/FLIO_Data', line, 'FLIO_parameters'))
            for file in files:
                if 'result' in file:
                    print(file)
                    shutil.copy(os.path.join(flimx_path, line.replace('_',  ''), file),
                                os.path.join('OakFundus/FLIO_Data', line, 'FLIO_parameters'))
        except:
            print(line + ' not found')