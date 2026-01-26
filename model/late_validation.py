import argparse
import os
import sys
import json
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor

# Ensure we can import CMliR from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.SCMliR import SCMLIR

def parse_args():
    parser = argparse.ArgumentParser(description="Late Validation for CMLIR: Select Best Caption based on Image-Text Similarity")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to CMLIR checkpoint (.pth)')
    parser.add_argument('--raw_csv', type=str, required=True, help='Path to raw input csv (must contain ID, Caption)')
    parser.add_argument('--rag_csv', type=str, required=True, help='Path to rag input csv (must contain ID, Caption)')
    parser.add_argument('--gt_csv', type=str, required=True, help='Path to Ground Truth CSV (must contain ID, Caption)')
    parser.add_argument('--annotation', type=str, required=True, help='Path to ground truth/annotation file (json) containing id and image_path')
    parser.add_argument('--image_root', type=str, default="", help='Root directory for images if paths in annotation are relative')
    parser.add_argument('--output_csv', type=str, default='late_validation.csv', help='Output CSV filename')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(
        args.checkpoint, 
        map_location=device, 
        weights_only=False
    )
    
    config = checkpoint["config"]
    if isinstance(config, dict):
        cfg = argparse.Namespace(**config)
    else:
        cfg = config
    
    if hasattr(cfg, 'delta_file'):
        cfg.delta_file = None

    cfg.RAG_prompt = False
    print(cfg)

    model = SCMLIR(cfg)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    # Load Image Processor
    print(f"Loading image processor for {model.hparams.vision_model}...")
    processor = AutoImageProcessor.from_pretrained(model.hparams.vision_model)

    # 2. Load Ground Truth Data (Text from CSV, Images from JSON)
    print(f"Loading Ground Truth Text from {args.gt_csv}...")
    df_gt = pd.read_csv(args.gt_csv)
    df_gt['ID'] = df_gt['ID'].astype(str)
    
    # 构建 ID -> Caption 的映射 (来自 CSV 的纯净文本)
    gt_text_map = dict(zip(df_gt['ID'], df_gt['Caption']))

    print(f"Loading Image Paths from {args.annotation}...")
    gt_data = {} # ID -> {image_paths: [], report: str}
    
    with open(args.annotation, 'r') as f:
        full_meta = json.load(f)
        
        if isinstance(full_meta, dict) and 'test' in full_meta:
            items = full_meta['test']
        else:
            print("Warning: No 'test' split found, assuming list structure.")
            items = full_meta
            
        count_matched = 0
        for item in items:
            item_id = str(item.get('id'))
            
            # 如果这个 ID 不在 GT CSV 里，可能是一个不需要评估的样本，跳过
            if item_id not in gt_text_map:
                continue

            paths = item.get('image_path', [])
            if isinstance(paths, str): paths = [paths]
            if not paths: continue
            
            full_img_paths = [os.path.join(args.image_root, p) for p in paths]
            
            # [关键修改] 使用 CSV 中的文本作为 Report，使用 JSON 中的路径作为 Image Path
            report = gt_text_map[item_id]
            
            gt_data[item_id] = {'image_paths': full_img_paths, 'report': report}
            count_matched += 1
            
        print(f"Loaded {count_matched} items with both Images (from JSON) and GT Text (from CSV).")

    # 3. Load Candidate CSVs
    print("Loading Candidate CSVs...")
    df_raw = pd.read_csv(args.raw_csv)
    df_rag = pd.read_csv(args.rag_csv)
    
    df_raw['ID'] = df_raw['ID'].astype(str)
    df_rag['ID'] = df_rag['ID'].astype(str)
    
    # Merge
    df_merged = pd.merge(df_raw, df_rag, on='ID', suffixes=('_raw', '_rag'))
    print(f"Merged Candidate samples: {len(df_merged)}")

    # 4. Validation Loop
    results = []
    print("Starting Late Validation (Similarity Scoring)...")
    
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged)):
        item_id = row['ID']
        raw_cap = str(row['Caption_raw'])
        rag_cap = str(row['Caption_rag'])
        
        if item_id not in gt_data:
            # 如果不在 GT 数据中，无法加载图片，只能跳过或标记
            continue
            
        full_img_paths = gt_data[item_id]['image_paths']
        
        # Load Images
        pixel_values_list = []
        try:
            for p in full_img_paths:
                image = Image.open(p).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                pixel_values_list.append(inputs.pixel_values.to(device))
        except Exception as e:
            print(f"Error loading images for {item_id}: {e}")
            continue
            
        if not pixel_values_list:
            continue

        # Compute Similarity
        with torch.no_grad():
            # Encode Image
            img_embeds, _ = model.encode_img(pixel_values_list) # (1, L, H_llama)
            
            # Project Image
            img_tok_low = F.normalize(model.img_shared_proj(img_embeds), dim=-1) # (1, L, 128)
            w_i2t = model.get_semantic_weights(img_embeds) # (1, L)
            
            # Encode Texts (Raw vs Rag)
            candidates = [raw_cap, rag_cap]
            t_inputs = model.medcpt_tokenizer(candidates, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            t_outputs = model.medcpt_model(**t_inputs)
            t_seq_768 = F.normalize(t_outputs.last_hidden_state, dim=-1)
            t_seq_low = F.normalize(model.txt_shared_proj(t_seq_768), dim=-1) # (2, L_txt, 128)
            
            # Expand Image to match candidates (2)
            img_tok_exp = img_tok_low.expand(2, -1, -1)
            w_i2t_exp = w_i2t.expand(2, -1)
            
            # Compute Interaction
            sim_matrix = model.compute_standard_late_interaction(img_tok_exp, t_seq_low, q_weights=w_i2t_exp, temperature=0.05)
            scores = torch.diag(sim_matrix).cpu().tolist() # [score_raw, score_rag]
            
            # [Selection Logic] 选择分数高的 Caption
            if scores[1] > scores[0]:
                results.append({'ID': item_id, 'Caption': rag_cap, 'Source': 'rag', 'Score': scores[1]})
            else:
                results.append({'ID': item_id, 'Caption': raw_cap, 'Source': 'raw', 'Score': scores[0]})

    # 5. Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output_csv, index=False)
    print(f"Saved selected captions to {args.output_csv}")
    
    # 6. Compute Metrics
    print("Computing metrics against Ground Truth (from CSV)...")
    
    hypo = {str(r['ID']): [r['Caption']] for r in results}
    # 这里使用的 report 是前面从 GT CSV 加载的
    ref = {str(k): [v['report']] for k, v in gt_data.items() if str(k) in hypo}
    
    # 确保只计算都有的样本
    common_ids = set(hypo.keys()) & set(ref.keys())
    hypo = {k: hypo[k] for k in common_ids}
    ref = {k: ref[k] for k in common_ids}
    
    print(f"Evaluating on {len(common_ids)} samples.")

    final_scores = model.score(ref, hypo)
    print("\nFinal Evaluation Scores:")
    print(json.dumps(final_scores, indent=4))
    
if __name__ == "__main__":
    main()