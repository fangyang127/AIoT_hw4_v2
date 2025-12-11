import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import PIL.Image as Image
import gradio as gr
from myna.augmentation import random_augment

# 辨識類別
category_en = "crested_myna,javan_myna,common_myna"

# 辨識類別的中文, 顯示時用的名稱
category_zh = "土八哥,白尾八哥,家八哥"

# APP 的名稱
title = "八哥辨識器"

# APP 的說明
description="請輸入一張八哥照片, 我會告訴你是什麼八哥!"

categories = category_en.split(',')
labels = category_zh.split(',')

# 辨識有幾類
N = len(categories)

# 讀取圖片資料
base_dir = 'myna/'
thedir = base_dir + categories[0]
data = []
target = []

# Data augmentation 開關與參數
AUGMENT = True
# 每張影像要產生幾個擴增版本（若不需要可設為 0）
AUGMENT_COUNT = 2

# 讀取所有圖片並轉成 NumPy 陣列
for i in range(N):
    thedir = base_dir + categories[i]
    file_names = os.listdir(thedir)
    for fname in file_names:
        img_path = thedir + '/' + fname
        img = load_img(img_path , target_size = (224,224))
        x = img_to_array(img)
        data.append(x)
        target.append(i)

        # 產生擴增影像並加入訓練資料
        if AUGMENT and AUGMENT_COUNT > 0:
            for _ in range(AUGMENT_COUNT):
                try:
                    aug_img = random_augment(img)
                    # 確保回傳的是 224x224
                    aug_img = aug_img.resize((224, 224), Image.Resampling.LANCZOS)
                    aug_arr = img_to_array(aug_img)
                    data.append(aug_arr)
                    target.append(i)
                except Exception:
                    # 若某些擴增失敗則跳過
                    pass

data = np.array(data)

# 將圖片資料做前處理
x_train = preprocess_input(data)

y_train = to_categorical(target, N)

# 建立模型
resnet = ResNet50V2(include_top=False, pooling="avg")
# 建立序列模型
model = Sequential()
# 加入 ResNet50V2 作為特徵擷取器
model.add(resnet)
# 加入輸出層
model.add(Dense(N, activation='softmax'))
#凍結 ResNet50V2 的權重
resnet.trainable = False

# 顯示模型摘要
model.summary()

# 編譯模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, batch_size=10, epochs=10)

# 評估模型
loss, acc = model.evaluate(x_train, y_train)
print(f"Loss: {loss}")
print(f"Accuracy: {acc}")

# 儲存完整模型，供後續推論使用
try:
    model.save('myna_model.h5')
    print('Model saved to myna_model.h5')
except Exception as e:
    print('Failed to save model:', e)

y_predict = np.argmax(model.predict(x_train), -1)

def resize_image(inp):
    # 將 NumPy array 轉換成 PIL Image 對象
    img = Image.fromarray(inp)

    # 將圖片調整為 224x224 像素
    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)

    # 將調整大小後的圖片轉換回 NumPy array
    img_array = np.array(img_resized)

    return img_array

def classify_image(inp):
    img_array = resize_image(inp)
    inp = img_array.reshape((1, 224, 224, 3))
    inp = preprocess_input(inp)
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(N)}

image = gr.Image(label="八哥照片")
label = gr.Label(num_top_classes=N, label="AI辨識結果")

sample_images = []
for i in range(N):
    thedir = base_dir + categories[i]
    for fname in os.listdir(thedir):
        sample_images.append(categories[i] + '/' + fname)

gr.Interface(fn=classify_image,
             inputs=image,
             outputs=label,
             title=title,
             description=description,
             examples=sample_images).launch(debug=True, share=True)