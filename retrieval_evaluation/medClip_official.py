import json
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import gc

# 导入官方 MedCLIP 库
from medclip import MedCLIPModel, MedCLIPProcessor

# ========= 工具函数 =========
def get_image_path(data_root, image_id):
    if "_train_" in image_id:
        sub_dir = "train"
    elif "_valid_" in image_id:
        sub_dir = "valid"
    elif "_test_" in image_id:
        sub_dir = "test"
    else:
        # 如果 ID 格式不含 train/valid/test，尝试直接搜索
        return os.path.join(data_root, f"{image_id}.jpg")
    return os.path.join(data_root, sub_dir, f"{image_id}.jpg")

def load_all_captions(data_root):
    caption_files = ["train_captions.csv", "valid_captions.csv", "test_captions.csv"]
    all_captions = {}
    for fname in caption_files:
        path = os.path.join(data_root, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.upper() for c in df.columns]
        for _, row in df.iterrows():
            all_captions[str(row["ID"])] = str(row["CAPTION"])
    return all_captions

# ========= 主流程 =========
def compute_medclip_official_similarities(
    json_path,
    data_root,
    output_dir,
    batch_size=64,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img2img_output = os.path.join(output_dir, "medclip_official_img2img.json")
    img2txt_output = os.path.join(output_dir, "medclip_official_img2txt.json")

    # ===== 1. 加载数据 =====
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions_dict = load_all_captions(data_root)

    # ===== 2. 加载官方 MedCLIP 模型与处理器 =====
    print("Loading official MedCLIP model and processor...")
    # 默认加载 medclip-vit 权重，你也可以指定 checkpoint 路径
    model = MedCLIPModel(vision_checkpoint=None)
    model.from_pretrained() # 自动下载官方在 HF 托管的权重
    model = model.to(device)
    model.eval()
    
    processor = MedCLIPProcessor()

    # ===== 3. 收集所有 ID =====
    unique_ids = set()
    target_ids_all = set()
    for item in data:
        unique_ids.add(item["id"])
        unique_ids.add(item["gold"])
        target_ids_all.add(item["gold"])
        for cand in item["candidates"]:
            unique_ids.add(cand)
            target_ids_all.add(cand)

    # ===== 4. Image 特征提取 (PyTorch) =====
    image_features_dict = {}
    print("\n[Step 4] Extracting image features...")
    
    with torch.no_grad():
        for img_id in tqdm(unique_ids):
            img_path = get_image_path(data_root, img_id)
            if not os.path.exists(img_path):
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                # 官方 Processor 处理图像
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # 提取并归一化特征
                img_embeds = model.encode_image(inputs['pixel_values'])
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                
                image_features_dict[img_id] = img_embeds.cpu().numpy()[0]
            except Exception as e:
                print(f"[Image Error] {img_id}: {e}")

    # ===== 5. Text 特征提取 (Batch) =====
    text_features_dict = {}
    target_ids_list = list(target_ids_all)
    print("\n[Step 5] Extracting text features...")

    with torch.no_grad():
        for i in tqdm(range(0, len(target_ids_list), batch_size)):
            batch_ids = target_ids_list[i : i + batch_size]
            batch_texts = [str(captions_dict.get(tid, "")) for tid in batch_ids]

            # 过滤掉空描述
            valid_indices = [j for j, txt in enumerate(batch_texts) if txt.strip()]
            if not valid_indices: continue
            
            curr_texts = [batch_texts[j] for j in valid_indices]
            curr_ids = [batch_ids[j] for j in valid_indices]

            try:
                # 官方 Processor 处理文本
                inputs = processor(text=curr_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                
                # 提取并归一化特征
                text_embeds = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                for tid, f in zip(curr_ids, text_embeds.cpu().numpy()):
                    text_features_dict[tid] = f
            except Exception as e:
                print(f"[Text Error] Batch {i}: {e}")

    # ===== 6. 相似度计算 =====
    i2i_res, i2t_res = [], []
    print("\n[Step 6] Computing similarities...")

    for item in tqdm(data):
        qid = item["id"]
        if qid not in image_features_dict:
            continue

        q_feat = image_features_dict[qid]
        targets = [item["gold"]] + item["candidates"]
        s_i2i, s_i2t = {}, {}

        for tid in targets:
            # I2I: 向量点积（因为已归一化，即余弦相似度）
            if tid in image_features_dict:
                s_i2i[tid] = round(float(np.dot(q_feat, image_features_dict[tid])), 4)
            else:
                s_i2i[tid] = 0.0

            # I2T
            if tid in text_features_dict:
                s_i2t[tid] = round(float(np.dot(q_feat, text_features_dict[tid])), 4)
            else:
                s_i2t[tid] = 0.0

        i2i_res.append({"id": qid, "scores": s_i2i})
        i2t_res.append({"id": qid, "scores": s_i2t})

    # ===== 7. 保存结果 =====
    with open(img2img_output, "w") as f:
        json.dump(i2i_res, f, indent=4)
    with open(img2txt_output, "w") as f:
        json.dump(i2t_res, f, indent=4)

    print(f"\n✅ Done. Official MedCLIP results saved to: {output_dir}")

# ========= 入口 =========
if __name__ == "__main__":
    # 请根据你的实际路径修改
    DATA_ROOT = "/home/users/h/hej/scratch/dataset/rocov2"
    JSON_INPUT = "rocov2_48x10_kis_benchmark.json"
    OUTPUT_DIR = "prediction"

    compute_medclip_official_similarities(
        JSON_INPUT,
        DATA_ROOT,
        OUTPUT_DIR,
        batch_size=64,
    )