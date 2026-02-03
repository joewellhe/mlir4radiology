import json
import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def get_image_path(data_root, image_id):
    """根据 ID 规则映射图片文件路径"""
    if "_train_" in image_id:
        sub_dir = "train"
    elif "_valid_" in image_id:
        sub_dir = "valid"
    elif "_test_" in image_id:
        sub_dir = "test"
    else:
        raise ValueError(f"Unknown image ID format: {image_id}")
    return os.path.join(data_root, sub_dir, f"{image_id}.jpg")

def load_all_captions(data_root):
    """从三个 CSV 文件加载所有 Caption"""
    caption_files = ["train_captions.csv", "test_captions.csv", "valid_captions.csv"]
    all_captions = {}
    print("Loading captions from CSV files...")
    for file_name in caption_files:
        file_path = os.path.join(data_root, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 假设 CSV 格式为 ID, Caption
            for _, row in df.iterrows():
                all_captions[str(row['ID'])] = str(row['Caption'])
        else:
            print(f"Warning: {file_path} not found.")
    return all_captions

def compute_clip_dual_similarities(json_path, data_root, output_dir, model_name="openai/clip-vit-base-patch32"):
    # 0. 准备工作
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img2img_output = os.path.join(output_dir, "clip_img2img_result.json")
    img2txt_output = os.path.join(output_dir, "clip_img2txt_result.json")

    # 1. 加载数据和 Caption
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions_dict = load_all_captions(data_root)

    # 2. 初始化 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 3. 收集所有唯一 ID
    unique_ids = set()
    target_ids_all = set() # 包含 gold 和 candidates，用于文本特征提取
    for item in data:
        unique_ids.add(item['id'])
        unique_ids.add(item['gold'])
        target_ids_all.add(item['gold'])
        for cand in item['candidates']:
            unique_ids.add(cand)
            target_ids_all.add(cand)
    
    print(f"Total unique images/texts to process: {len(unique_ids)}")

    # 4. 预计算所有图片的 Feature (Image Embedding)
    image_features_dict = {}
    model.eval()
    with torch.no_grad():
        for img_id in tqdm(unique_ids, desc="Extracting image features"):
            img_path = get_image_path(data_root, img_id)
            if not os.path.exists(img_path):
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                img_feat = model.get_image_features(**inputs)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                image_features_dict[img_id] = img_feat.cpu()
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")

    # 5. 预计算所有目标的文本 Feature (Text Embedding)
    text_features_dict = {}
    with torch.no_grad():
        for txt_id in tqdm(target_ids_all, desc="Extracting text features"):
            caption = captions_dict.get(txt_id)
            if not caption:
                continue
            try:
                # 注意：CLIP 有 77 tokens 的长度限制，processor 会自动处理截断
                inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True).to(device)
                txt_feat = model.get_text_features(**inputs)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                text_features_dict[txt_id] = txt_feat.cpu()
            except Exception as e:
                print(f"Error processing text {txt_id}: {e}")

    # 6. 计算相似度并构造输出
    img2img_results = []
    img2txt_results = []

    for item in tqdm(data, desc="Calculating similarities"):
        query_id = item['id']
        if query_id not in image_features_dict:
            continue
            
        query_img_feat = image_features_dict[query_id]
        target_ids = [item['gold']] + item['candidates']
        
        img2img_scores = {}
        img2txt_scores = {}
        
        for tid in target_ids:
            # 图图相似度 (Query Image vs Target Image)
            if tid in image_features_dict:
                sim_ii = torch.matmul(query_img_feat, image_features_dict[tid].T).item()
                img2img_scores[tid] = round(float(sim_ii), 4)
            else:
                img2img_scores[tid] = 0.0

            # 图文相似度 (Query Image vs Target Text)
            if tid in text_features_dict:
                sim_it = torch.matmul(query_img_feat, text_features_dict[tid].T).item()
                img2txt_scores[tid] = round(float(sim_it), 4)
            else:
                img2txt_scores[tid] = 0.0
        
        img2img_results.append({"id": query_id, "scores": img2img_scores})
        img2txt_results.append({"id": query_id, "scores": img2txt_scores})

    # 7. 保存两个结果文件
    with open(img2img_output, 'w', encoding='utf-8') as f:
        json.dump(img2img_results, f, indent=4, ensure_ascii=False)
    with open(img2txt_output, 'w', encoding='utf-8') as f:
        json.dump(img2txt_results, f, indent=4, ensure_ascii=False)
    
    print(f"Done!\nImage-to-Image saved to: {img2img_output}\nImage-to-Text saved to: {img2txt_output}")

if __name__ == "__main__":
    DATA_ROOT = "/home/users/h/hej/scratch/dataset/rocov2"
    JSON_INPUT = "rocov2_48x10_kis_benchmark.json"
    OUTPUT_DIR = "prediction"

    compute_clip_dual_similarities(JSON_INPUT, DATA_ROOT, OUTPUT_DIR)