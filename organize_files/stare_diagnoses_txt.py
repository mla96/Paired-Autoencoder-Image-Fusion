# From diagnoses .txt, get list of AMD or drusen images, then move them into a separate folder


path = '../../../../OakData/Transfer_learning_datasets/STARE/diagnoses.txt'
new_file = '../../../../OakData/Transfer_learning_datasets/STARE/AMD_diag_subjs_list.txt'
new_file_list = []

f = open(path, 'r')
for line in f.readlines():
    line = line.strip()
    if 'Drusen' in line or 'Age Related' in line:
        new_file_list.append(line)
        print(line)
f.close()

with open(new_file, "w") as txt_file:
    for line in new_file_list:
        line = line.split('\t')[0]
        txt_file.write(line + "\n")
        print(line)


