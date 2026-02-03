import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc

# ========= 环境变量：防止 JAX 吃满显存 =========
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

from transformers import CLIPProcessor, AutoTokenizer
from medclip.modeling_hybrid_clip import FlaxHybridCLIP


# ========= 工具函数 =========
def get_image_path(data_root, image_id):
    if "_train_" in image_id:
        sub_dir = "train"
    elif "_valid_" in image_id:
        sub_dir = "valid"
    elif "_test_" in image_id:
        sub_dir = "test"
    else:
        raise ValueError(f"Unknown ID: {image_id}")
    return os.path.join(data_root, sub_dir, f"{image_id}.jpg")


def load_all_captions(data_root):
    caption_files = [
        "train_captions.csv",
        "valid_captions.csv",
        "test_captions.csv",
    ]
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
def compute_medclip_dual_similarities(
    json_path,
    data_root,
    output_dir,
    batch_size=64,
):
    os.makedirs(output_dir, exist_ok=True)

    img2img_output = os.path.join(output_dir, "medclip_img2img_result.json")
    img2txt_output = os.path.join(output_dir, "medclip_img2txt_result.json")

    # ===== 1. 加载数据 =====
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions_dict = load_all_captions(data_root)

    # ===== 2. 加载 MedCLIP =====
    model_id = "flax-community/medclip-roco"
    print(f"Loading MedCLIP model: {model_id}")
    model = FlaxHybridCLIP.from_pretrained(model_id)

    # Vision processor（CLIP ViT-B/32）
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Text tokenizer（SciBERT，与 MedCLIP 文本塔一致）
    text_tokenizer = AutoTokenizer.from_pretrained(
        "allenai/scibert_scivocab_uncased"
    )

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

    # ===== 4. Image 特征提取 =====
    image_features_dict = {}
    print("\n[Step 4] Extracting image features...")

    for img_id in tqdm(unique_ids):
        img_path = get_image_path(data_root, img_id)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = image_processor(images=image, return_tensors="np")

            pixel_values = inputs["pixel_values"]          # (1, 3, H, W)
            pixel_values = np.transpose(
                pixel_values, (0, 2, 3, 1)
            ).astype(np.float32)                           # 🔧 FIX: NHWC + float32

            outputs = model.get_image_features(pixel_values)
            feat = np.array(outputs)[0].astype(np.float32)  # 🔧 FIX: 不 flatten batch

            norm = np.linalg.norm(feat)
            if norm > 1e-12:
                feat = feat / norm

            image_features_dict[img_id] = np.nan_to_num(feat)

        except Exception as e:
            print(f"[Image Error] {img_id}: {e}")

    gc.collect()

    # ===== 5. Text 特征提取（Batch）=====
    text_features_dict = {}
    target_ids_list = list(target_ids_all)

    print("\n[Step 5] Extracting text features (SciBERT, batched)...")

    for i in tqdm(range(0, len(target_ids_list), batch_size)):
        batch_ids = target_ids_list[i : i + batch_size]
        batch_texts = [str(captions_dict.get(tid, "")) for tid in batch_ids]

        valid_idx = [j for j, t in enumerate(batch_texts) if t.strip()]
        if not valid_idx:
            continue

        curr_ids = [batch_ids[j] for j in valid_idx]
        curr_texts = [batch_texts[j] for j in valid_idx]

        try:
            inputs = text_tokenizer(
                curr_texts,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=128,
                return_token_type_ids=True,    
            )

            outputs = model.get_text_features(**inputs)
            feats = np.array(outputs).astype(np.float32)

            feats = np.nan_to_num(feats)
            norms = np.linalg.norm(feats, axis=-1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            feats = feats / norms

            for tid, f in zip(curr_ids, feats):
                text_features_dict[tid] = f

        except Exception as e:
            print(f"[Text Error] batch {i}: {e}")

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
            # Image → Image
            if tid in image_features_dict:
                s_i2i[tid] = round(
                    float(np.dot(q_feat, image_features_dict[tid])), 4
                )
            else:
                s_i2i[tid] = 0.0

            # Image → Text
            if tid in text_features_dict:
                s_i2t[tid] = round(
                    float(np.dot(q_feat, text_features_dict[tid])), 4
                )
            else:
                s_i2t[tid] = 0.0

        i2i_res.append({"id": qid, "scores": s_i2i})
        i2t_res.append({"id": qid, "scores": s_i2t})

    # ===== 7. 保存结果 =====
    with open(img2img_output, "w") as f:
        json.dump(i2i_res, f, indent=4)

    with open(img2txt_output, "w") as f:
        json.dump(i2t_res, f, indent=4)

    print(f"\n✅ Done. Results saved to: {output_dir}")


# ========= 入口 =========
if __name__ == "__main__":
    DATA_ROOT = "/home/users/h/hej/scratch/dataset/rocov2"
    JSON_INPUT = "rocov2_48x10_kis_benchmark.json"
    OUTPUT_DIR = "prediction"

    compute_medclip_dual_similarities(
        JSON_INPUT,
        DATA_ROOT,
        OUTPUT_DIR,
        batch_size=64,
    )
