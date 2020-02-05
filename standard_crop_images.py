import os
from PIL import Image


os.getcwd()
os.chdir("../")

# print(os.getcwd())

path = 'STARE_Data'

files = os.listdir(path)
# f = open(path, 'r')
# lines = f.readlines()
# f.close()

crop_bound_w = 100  # Pixel length of cropping for width and height boundaries on fundus images
crop_bound_h = 50

final_dim = 1612

quality_val = 90


for file in files:
    file = file.strip()

    file_image = Image.open(os.path.join(path, file))  # Load image

    # Resize and format fixed image (fundus RGB image) into BW SITK image for registration
    w, h = file_image.size
    file_image = file_image.crop((crop_bound_w, crop_bound_h, w - crop_bound_w, h - crop_bound_h))
    w, h = file_image.size
    file_image = file_image.resize((int(final_dim), int(final_dim * h / w)), Image.LANCZOS)
    w, h = file_image.size
    file_image = file_image.crop((0, (h - w)/2, w, h - (h - w)/2))

    filename_split = file.split('.')
    file_image.save(os.path.join(path, filename_split[0] + '_crop.' + filename_split[1]), 'JPEG', quality=quality_val)
