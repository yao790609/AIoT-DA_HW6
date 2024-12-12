# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:03:51 2024

@author: User
"""
import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import decode_predictions
import io
from PIL import Image

# 步驟 1: 載入 VGG16 預訓練模型
def build_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # 冻結 VGG16 的層
    for layer in base_model.layers:
        layer.trainable = False
    # 添加自定義層
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)  # 設置為2個類別
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 步驟 2: 用於訓練的醫療口罩資料集
# 假設您有一個訓練資料集，這裡示範如何進行訓練
# 這部分需要您自已准備一個醫療口罩資料集並進行訓練
def train_model(model, train_generator, validation_generator):
    model.fit(train_generator, epochs=10, validation_data=validation_generator)
    return model

# 步驟 3: 輸入圖片 URL 並進行分類
def test_image(image_url, model, class_names):
    # 下載圖片並進行預處理
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content)).resize((224, 224))
    img_array = np.array(img)  # 轉換為數組
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度
    img_array = preprocess_input(img_array)  # 預處理圖片

    # 預測圖片
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]

    print(f"The image is classified as: {predicted_class_name}")

# 假設您的類別是戴口罩 (0) 和未戴口罩 (1)
class_names = ["Mask", "No Mask"]

# 步驟 1: 構建 VGG16 模型
model = build_vgg16_model()

# 假設你有一個訓練資料集並且已經完成了訓練過程
# train_generator 和 validation_generator 是你訓練過程中的資料生成器
# model = train_model(model, train_generator, validation_generator)

# 步驟 3: 測試一張圖片
image_url = input("Please enter image URL: ")
test_image(image_url, model, class_names)
