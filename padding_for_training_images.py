import cv2
import os
import glob
import numpy as np
from PIL import Image

### 1. $ cd YoloTrainDataGenerate
### 2. $ sudo chmod +x *.sh
### 3. $ ./setup.sh
### 4. Copy the training image to "images_org" folder
###    (example) ./images_org/Person_0001.png
###              ./images_org/Person_0002.png
###              ./images_org/dog_0001.png
###              ./images_org/dog_0002.png
###              ./images_org/dog_0003.png
### 5. $ nano padding_for_training_images.py
### 6. Change the total number of images after padding
###    (example before) train_images = 100
###    (example after)  train_images = 1000
### 7. $ python3 padding_for_training_images.py
### 8. The padded image will be created directly under the "Images" folder


# Deduplication function in List
def remove_duplicates(x):
    y=[]
    for i in x:
        if i not in y:
            y.append(i)
    return y

# Histogram homogenization function
def equalizeHistRGB(src):
    
    RGB = cv2.split(src)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])
    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

# Gaussian noise function
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss
    return noisy

# salt & pepper noise function
def addSaltPepperNoise(src):
    row,col,ch = src.shape
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()

    # Salt mode
    try:
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i-1 , int(num_salt)) for i in src.shape]
        out[coords[:-1]] = (255,255,255)
    except:
        pass

    # Pepper mode
    try:
        num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in src.shape]
        out[coords[:-1]] = (0,0,0)
    except:
        pass
    return out


base_path = os.getcwd()
img_files = glob.glob("images_org/*")
imgs = []
labels = []
labelsdist = []
for img_file in img_files:
    labels.append(img_file.split("/")[-1].split(".")[0].split("_")[0])
    imgs.append(cv2.imread(img_file, cv2.IMREAD_UNCHANGED))
labelsdist = remove_duplicates(labels)

# write label file
with open("label.txt", "w") as f:
    for label in labelsdist:
        f.write("%s\n" % (label))

######################################################
train_images = 100
######################################################

# Generate lookup table
min_table = 50
max_table = 205
diff_table = max_table - min_table
gamma1 = 0.75
gamma2 = 1.5
LUT_HC = np.arange(256, dtype = 'uint8')
LUT_LC = np.arange(256, dtype = 'uint8')
LUT_G1 = np.arange(256, dtype = 'uint8')
LUT_G2 = np.arange(256, dtype = 'uint8')
LUTs = []
# Smoothing sequence
average_square = (10,10)
# Create high contrast LUT
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table                        
for i in range(max_table, 255):
    LUT_HC[i] = 255
# Other LUT creation
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
LUTs.append(LUT_HC)
LUTs.append(LUT_LC)
LUTs.append(LUT_G1)
LUTs.append(LUT_G2)

for i in range(train_images):

    class_id = np.random.randint(len(labels))
    img = imgs[class_id]
    
    # Contrast conversion execution
    if np.random.randint(2) == 1:
        level = np.random.randint(4)
        img = cv2.LUT(img, LUTs[level])

    # Smoothing execution
    if np.random.randint(2) == 1:
        img = cv2.blur(img, average_square)

    # Histogram equalization execution
    if np.random.randint(2) == 1:
        img = equalizeHistRGB(img)

    # Gaussian noise addition execution
    if np.random.randint(2) == 1:
        img = addGaussianNoise(img)

    # Salt & Pepper noise addition execution
    if np.random.randint(2) == 1:
        img = addSaltPepperNoise(img)

    image_path = "%s/images/%s_%s.png" % (base_path, labels[class_id], i)
    cv2.imwrite(image_path, img)

    print("train image", i, labels[class_id])


