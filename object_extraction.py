import numpy as np
import cv2
import os
import math

# 画像入出力先パス、拡張子
# EXTには処理対象とする画像ファイルの拡張子を指定
INP = "./images/"
EXT = ".png"

# 背景全面白色塗りつぶし
def mask_back_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_min = np.array([0, 0, 50], np.uint8)
    white_max = np.array([180, 39, 255], np.uint8)
    white_region = cv2.inRange(hsv, white_min, white_max)
    white = np.full(img.shape, 255, dtype=img.dtype)
    background = cv2.bitwise_and(white, white, mask=white_region)
    inv_mask = cv2.bitwise_not(white_region)
    extracted = cv2.bitwise_and(img, img, mask=inv_mask)
    masked = cv2.add(extracted, background)
    return masked

# 物体部分全面白色塗りつぶし
def mask_front_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_min = np.array([0, 0, 1], np.uint8)
    white_max = np.array([255, 255, 255], np.uint8)
    white_region = cv2.inRange(hsv, white_min, white_max)
    white = np.full(img.shape, 255, dtype=img.dtype)
    background = cv2.bitwise_and(white, white, mask=white_region)
    inv_mask = cv2.bitwise_not(white_region)
    extracted = cv2.bitwise_and(img, img, mask=inv_mask)
    masked = cv2.add(extracted, background)
    return masked

# ノイズ除去
def morphology(img):
    kernel = np.ones((12,12),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  
    return opening

# 物体検出
def detect_contour(img, min_size):
    contoured = img.copy()
    forcrop = img.copy()
    # グレースケール画像生成
    dst = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2GRAY)
    # 輪郭検出
    im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    # 輪郭抽出
    for c in contours:
        # min_sizeに満たない面積の物体は無視
        if cv2.contourArea(c) < min_size:
            continue
        # バウンディングボックスサイズの検出とパディング追加調整
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = padding_position(x, y, w, h, 5)
        # 物体部の画像切り取り
        if y < 0:
            h = h + y
            y = 0
        if x < 0:
            w = w + x
            x = 0
        cropped = forcrop[y:(y + h), x:(x + w)]
        cropped = resize_image(cropped, (96, 96))
        crops.append(cropped)
        # 輪郭・バウンディングボックス描写
        cv2.drawContours(contoured, c, -1, (0, 0, 255), 3)
        cv2.rectangle(contoured, (x, y), (x + w, y + h), (0, 255, 0), 7)
    return contoured, crops

# パディング調整
def padding_position(x, y, w, h, p):
    return x - p, y - p, w + p * 2, h + p * 2

# 画像リサイズ 2018.02.12 回転時に見切れが発生するバグ修正
def resize_image(img, size):
    img_size = img.shape[:2]
    newheight = img_size[0]
    newwidth  = img_size[1]
    # サイズ縮小
    if newheight > size[1] or newwidth > size[0]:
        if newheight > newwidth:
            newheight = int(round(math.sqrt(size[0] ** 2 + size[1] ** 2) / 2))
            raito =  newheight / img_size[0]
            newwidth = int(img_size[1] * raito)
        else:
            newwidth = int(round(math.sqrt(size[0] ** 2 + size[1] ** 2) / 2))
            raito = newwidth / img_size[1] 
            newheight = int(img_size[0] * raito)
        img = cv2.resize(img, (newwidth, newheight))
        img_size = img.shape[:2]
    # 画像のセンタリング
    row = (size[1] - newheight) // 2
    col = (size[0] - newwidth) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + newheight), col:(col + newwidth)] = img
    return resized

############ メイン処理部 ##############

# ファイル名の取得
fileList = os.listdir(INP)

for file in fileList:
    # ファイル拡張子の抽出
    base, ext = os.path.splitext(file)
    # 処理対象の拡張子のみ処理実施
    if ext == EXT:
        # 元画像読み込み
        filename = INP + file
        img = cv2.imread(filename)
        # 元画像背景白色化
        img = mask_back_white(img)
        # 元画像ノイズ除去
        #img = morphology(img)
        # 物体エリア検出
        contoured, crops = detect_contour(img, 1000)
        # 背景透過処理
        for i, c in enumerate(crops):
            # 背景黒色化（元画像のマスク用画像）
            lower = np.array([255,255,255])
            upper = np.array([255,255,255])
            img_mask = cv2.inRange(c, lower, upper)
            img_mask = cv2.bitwise_not(img_mask,img_mask)
            img_mask = cv2.bitwise_and(c, c, mask=img_mask)
            # 物体白色化（元画像のマスク用画像）
            img_mask = mask_front_white(img_mask)
            # マスク用画像のグレースケール変換（２値化）
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            # 元画像とマスク用画像の合成
            img = cv2.split(c)
            img = cv2.merge(img + [img_mask])
            mono = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0].mean()
            # 無地の画像は出力しない判定
            if mono != 0:
                # 加工完了画像ファイル出力
                OUT = INP + "_" + base + "-" + str(("%04d" % i)) + ext
                cv2.imwrite(OUT, img)

