# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:11:59 2024

@author: User
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

# 設定資料夾路徑
with_mask_dir = r"C:\Users\User\Downloads\Face-Mask-Detection-master\Face-Mask-Detection-master\dataset\with_mask"
without_mask_dir = r"C:\Users\User\Downloads\Face-Mask-Detection-master\Face-Mask-Detection-master\dataset\without_mask"

# 定義圖像轉換操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 自定義圖像數據集
class CustomImageDataset(Dataset):
    def __init__(self, with_mask_dir, without_mask_dir, transform=None):
        self.with_mask_dir = with_mask_dir
        self.without_mask_dir = without_mask_dir
        self.transform = transform

        # 獲取有口罩和無口罩圖像的路徑
        self.with_mask_images = [os.path.join(with_mask_dir, f) for f in os.listdir(with_mask_dir)]
        self.without_mask_images = [os.path.join(without_mask_dir, f) for f in os.listdir(without_mask_dir)]

        # 標籤：1代表有口罩，0代表無口罩
        self.labels = [1] * len(self.with_mask_images) + [0] * len(self.without_mask_images)

        # 合併圖像路徑
        self.image_paths = self.with_mask_images + self.without_mask_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 根據索引加載圖像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        # 如果有轉換操作，則應用它
        if self.transform:
            image = self.transform(image)

        # 返回字典而非元組
        return {
            'pixel_values': image,  # 圖像的張量
            'labels': torch.tensor(label, dtype=torch.long)  # 標籤
        }

# 載入數據集
dataset = CustomImageDataset(with_mask_dir, without_mask_dir, transform)

# 將數據集分割為訓練集和驗證集
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 將數據集轉換為Hugging Face Dataset格式
train_dataset = HFDataset.from_dict({
    'pixel_values': [x['pixel_values'] for x in train_data],
    'labels': [x['labels'] for x in train_data]
})

val_dataset = HFDataset.from_dict({
    'pixel_values': [x['pixel_values'] for x in val_data],
    'labels': [x['labels'] for x in val_data]
})

# 下載預訓練模型
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./results",           # 結果存放目錄
    evaluation_strategy="epoch",      # 訓練每個epoch後進行評估
    learning_rate=2e-5,               # 學習率
    per_device_train_batch_size=16,   # 訓練批次大小
    per_device_eval_batch_size=16,    # 驗證批次大小
    num_train_epochs=3,               # 訓練epochs數量
    weight_decay=0.01,                # 權重衰減
    logging_dir="./logs",             # 日誌目錄
)

# 定義Trainer
trainer = Trainer(
    model=model,                           # 使用的模型
    args=training_args,                    # 訓練參數
    train_dataset=train_dataset,           # 訓練數據集
    eval_dataset=val_dataset,              # 驗證數據集
)

# 開始訓練
trainer.train()
