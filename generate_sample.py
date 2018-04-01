import cv2
import os
import glob
import numpy as np
from PIL import Image

def overlay(src_image, overlay_image, pos_x, pos_y):
    ol_height, ol_width = overlay_image.shape[:2]

    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    src_image_PIL=Image.fromarray(src_image_RGBA)
    overlay_image_PIL=Image.fromarray(overlay_image_RGBA)

    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    tmp.paste(overlay_image_PIL, (pos_x, pos_y), overlay_image_PIL)
    result = Image.alpha_composite(src_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)

def delete_pad(image): 
    orig_h, orig_w = image.shape[:2]
    mask = np.argwhere(image[:, :, 3] > 128)
    (min_y, min_x) = (max(min(mask[:, 0])-1, 0), max(min(mask[:, 1])-1, 0))
    (max_y, max_x) = (min(max(mask[:, 0])+1, orig_h), min(max(mask[:, 1])+1, orig_w))
    return image[min_y:max_y, min_x:max_x]

def rotate_image(image, angle):
    orig_h, orig_w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((orig_h/2, orig_w/2), angle, 1)
    return cv2.warpAffine(image, matrix, (orig_h, orig_w))

def scale_image(image, scale):
    orig_h, orig_w = image.shape[:2]
    return cv2.resize(image, (int(orig_w*scale), int(orig_h*scale)))

def random_sampling(image, h, w): 
    orig_h, orig_w = image.shape[:2]
    y = np.random.randint(orig_h-h+1)
    x = np.random.randint(orig_w-w+1)
    return image[y:y+h, x:x+w]

def random_rotate_scale_image(image):
    image = rotate_image(image, np.random.randint(360))
    image = scale_image(image, 1 + np.random.rand() * 2)
    return delete_pad(image)

def random_overlay_image(src_image, overlay_image):
    src_h, src_w = src_image.shape[:2]
    overlay_h, overlay_w = overlay_image.shape[:2]
    y = np.random.randint(src_h-overlay_h+1)
    x = np.random.randint(src_w-overlay_w+1)
    bbox = ((x, y), (x+overlay_w, y+overlay_h))
    return overlay(src_image, overlay_image, x, y), bbox

def yolo_format_bbox(image, bbox):
    orig_h, orig_w = image.shape[:2]
    center_x = (bbox[1][0] + bbox[0][0]) / 2 / orig_w
    center_y = (bbox[1][1] + bbox[0][1]) / 2 / orig_h
    w = (bbox[1][0] - bbox[0][0]) / orig_w
    h = (bbox[1][1] - bbox[0][1]) / orig_h
    return(center_x, center_y, w, h)

# List内重複排除関数
def remove_duplicates(x):
    y=[]
    for i in x:
        if i not in y:
            y.append(i)
    return y

# ヒストグラム均一化関数
def equalizeHistRGB(src):
    
    RGB = cv2.split(src)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])
    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

# ガウシアンノイズ関数
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss
    return noisy

# salt&pepperノイズ関数
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
fruit_files = glob.glob("images_org/*")
fruits = []
labels = []
labelsdist = []
for fruit_file in fruit_files:
    labels.append(fruit_file.split("/")[-1].split(".")[0].split("_")[0])
    fruits.append(cv2.imread(fruit_file, cv2.IMREAD_UNCHANGED))
background_image = cv2.imread("background.jpg")
labelsdist = remove_duplicates(labels)

# write label file
with open("label.txt", "w") as f:
    for label in labelsdist:
        f.write("%s\n" % (label))

background_height, background_width = (416, 416)
train_images = 10000
test_images = 2000

# ルックアップテーブルの生成
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
# 平滑化用配列
average_square = (10,10)
# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table                        
for i in range(max_table, 255):
    LUT_HC[i] = 255
# その他LUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
LUTs.append(LUT_HC)
LUTs.append(LUT_LC)
LUTs.append(LUT_G1)
LUTs.append(LUT_G2)

for i in range(train_images):
    sampled_background = random_sampling(background_image, background_height, background_width)

    class_id = np.random.randint(len(labels))
    fruit = fruits[class_id]
    fruit = random_rotate_scale_image(fruit)

    result, bbox = random_overlay_image(sampled_background, fruit)
    yolo_bbox = yolo_format_bbox(result, bbox)

    # コントラスト変換実行
    if np.random.randint(2) == 1:
        level = np.random.randint(4)
        result = cv2.LUT(result, LUTs[level])

    # 平滑化実行
    if np.random.randint(2) == 1:
        result = cv2.blur(result, average_square)

    # ヒストグラム均一化実行
    if np.random.randint(2) == 1:
        result = equalizeHistRGB(result)

    # ガウシアンノイズ付加実行
    if np.random.randint(2) == 1:
        result = addGaussianNoise(result)

    # Salt & Pepperノイズ付加実行
    if np.random.randint(2) == 1:
        result = addSaltPepperNoise(result)

    # 反転実行
    if np.random.randint(2) == 1:
        result = cv2.flip(result, 1)

    image_path = "%s/images/train_%s_%s.jpg" % (base_path, i, labels[class_id])
    cv2.imwrite(image_path, result)

    with open("train.txt", "a") as f:
        f.write("%s\n" % (image_path))

    label_path = "%s/labels/train_%s_%s.txt" % (base_path, i, labels[class_id]) 
    with open(label_path, "w") as f:
        f.write("%s %s %s %s %s" % (labelsdist.index(labels[class_id]), yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3]))

    print("train image", i, labels[class_id], yolo_bbox)

for i in range(test_images):
    sampled_background = random_sampling(background_image, background_height, background_width)

    class_id = np.random.randint(len(labels))
    fruit = fruits[class_id]
    fruit = random_rotate_scale_image(fruit)

    result, bbox = random_overlay_image(sampled_background, fruit)
    yolo_bbox = yolo_format_bbox(result, bbox)

    # コントラスト変換実行
    if np.random.randint(2) == 1:
        level = np.random.randint(4)
        result = cv2.LUT(result, LUTs[level])

    # 平滑化実行
    if np.random.randint(2) == 1:
        result = cv2.blur(result, average_square)

    # ヒストグラム均一化実行
    if np.random.randint(2) == 1:
        result = equalizeHistRGB(result)

    # ガウシアンノイズ付加実行
    if np.random.randint(2) == 1:
        result = addGaussianNoise(result)

    # Salt & Pepperノイズ付加実行
    if np.random.randint(2) == 1:
        result = addSaltPepperNoise(result)

    # 反転実行
    if np.random.randint(2) == 1:
        result = cv2.flip(result, 1)

    image_path = "%s/images/test_%s_%s.jpg" % (base_path, i, labels[class_id])
    cv2.imwrite(image_path, result)

    with open("test.txt", "a") as f:
        f.write("%s\n" % (image_path))

    label_path = "%s/labels/test_%s_%s.txt" % (base_path, i, labels[class_id]) 
    with open(label_path, "w") as f:
        f.write("%s %s %s %s %s" % (labelsdist.index(labels[class_id]), yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3]))

    print("test image", i, labels[class_id], yolo_bbox)

