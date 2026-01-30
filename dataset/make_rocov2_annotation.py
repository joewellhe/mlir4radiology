import os
import json
import pandas as pd
import random
import math
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录名称
ROOT_DIR = "/home/users/h/hej/scratch/dataset/rocov2"

# 输入文件路径
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, "train_captions.csv")
VALID_CSV_PATH = os.path.join(ROOT_DIR, "valid_captions.csv")
TEST_CSV_PATH = os.path.join(ROOT_DIR, "test_captions.csv")

# 图片所在的子文件夹名称
TRAIN_IMG_DIR = "train"
VALID_IMG_DIR = "valid"
TEST_IMG_DIR = "test"

# 输出文件路径
OUTPUT_JSON_PATH = os.path.join(ROOT_DIR, "annotation.json")

# 随机种子，保证每次运行切分结果一致
RANDOM_SEED = 42
# ===========================================

def create_entry(row, img_folder, split_label):
    """
    辅助函数：将一行CSV数据转换为目标字典格式
    """
    img_id = str(row['ID'])
    caption = str(row['Caption'])
    
    # 构造图片路径，假设图片格式为 .jpg
    # 注意：这里保留了文件夹前缀，例如 "train/ImageID.jpg"
    # 这样在读取时只需要拼接 dataset root 即可
    img_filename = f"{img_id}.jpg"
    full_img_path = os.path.join(img_folder, img_filename)
    
    # 转换为 IU_xray 格式
    return {
        "id": img_id,
        "report": caption,
        "image_path": [full_img_path], # IU_xray 格式要求是列表
        "split": split_label
    }

def main():
    print(f"start processing {ROOT_DIR}: {ROOT_DIR}")
    
    # 1. 处理原始的 Train 数据
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"fail: connot find {TRAIN_CSV_PATH}")
        return

    print("reding Train CSV...")
    df_train_raw = pd.read_csv(TRAIN_CSV_PATH)
    
    # 将 DataFrame 转换为列表以便处理
    raw_train_items = []
    for _, row in df_train_raw.iterrows():
        # 暂时不标记 split，切分后再标记
        raw_train_items.append(row)
        
    print(f"Train count: {len(raw_train_items)}")
    
    
    # 2. 处理原始的 Valid 数据 
    if not os.path.exists(VALID_CSV_PATH):
        print(f"fail: connot find {VALID_CSV_PATH}")
        return

    print("reading Valid CSV (as Test Set)...")
    df_valid_raw = pd.read_csv(VALID_CSV_PATH)
    raw_valid_items = []
    for _, row in df_valid_raw.iterrows():
        raw_valid_items.append(row)

    print(f"Valid count: {len(raw_valid_items)}")

    # 3. test set
    if not os.path.exists(TEST_CSV_PATH):
        print(f"fail: connot find {TEST_CSV_PATH}")
        return
    
    print("reading Test CSV...")    
    df_test_raw = pd.read_csv(TEST_CSV_PATH)
    raw_test_items = []
    for _, row in df_test_raw.iterrows():
        raw_test_items.append(row)

    print(f"Test count: {len(raw_test_items)}")

    # 3. 构建最终的 JSON 结构
    final_annotation = {
        "train": [],
        "val": [],
        "test": []
    }

    # 填充 Train
    for row in tqdm(raw_train_items, desc="Processing Train Data"):
        entry = create_entry(row, TRAIN_IMG_DIR, "train")
        final_annotation["train"].append(entry)

    # 填充 Val
    for row in tqdm(raw_valid_items, desc="Processing Val Data"):
        entry = create_entry(row, VALID_IMG_DIR, "val")
        final_annotation["val"].append(entry)

    # 填充 Test
    for row in tqdm(raw_test_items, desc="Processing Test Data"):
        entry = create_entry(row, TEST_IMG_DIR, "test") 
        final_annotation["test"].append(entry)

    # 4. 写入文件
    print(f"writing {OUTPUT_JSON_PATH} ...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_annotation, f, ensure_ascii=False, indent=4)

    print("compelted!")

if __name__ == "__main__":
    main()
