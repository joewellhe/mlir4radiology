import json
import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor
import sys
# Ensure we can import CMliR from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.SCMliR import SCMLIR 

# ========= 工具函数 (独立于模型) =========
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

# ========= 评估器类 =========
class SCMLIREvaluator:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CMLIR model from {model_path}...")
        
        # 1. 加载 Checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 2. 从权重中提取配置
        self.args = checkpoint["config"]
        
        # 3. 初始化模型并加载权重
        print(self.args)
        self.args.delta_file = None  
        self.model = SCMLIR(self.args)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_img_feature(self, img_path, processor):
        """提取图像的序列特征和语义权重"""
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        # 视觉编码流水线
        swin_out = self.model.visual_encoder(inputs.pixel_values)['last_hidden_state']
        llama_space = self.model.layer_norm(self.model.llama_proj(swin_out))
        
        # 投影到检索空间 (Batch=1, Seq, 128) -> (Seq, 128)
        img_low = F.normalize(self.model.img_shared_proj(llama_space), dim=-1)[0]
        # 获取语义重要性权重 (Seq)
        weights = self.model.get_semantic_weights(llama_space)[0]
        
        return img_low, weights

    @torch.no_grad()
    def extract_txt_feature(self, text):
        """提取文本的序列特征"""
        t_inputs = self.model.medcpt_tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        t_outputs = self.model.medcpt_model(**t_inputs).last_hidden_state
        
        # --- 修复：添加训练时存在的第一次归一化 ---
        t_outputs = F.normalize(t_outputs, dim=-1) 
        # ---------------------------------------

        # 投影到检索空间
        # txt_low = F.normalize(self.model.txt_shared_proj(t_outputs), dim=-1)[0]
        txt_low = self.model.txt_shared_proj(t_outputs)[0]
        return txt_low

    def compute_scores(self, q_low, q_w, target_low):
        """调用模型内置的 Late Interaction 逻辑"""
        # unsqueeze(0) 是为了适配模型中 (Batch, Seq, Dim) 的输入要求
        return self.model.compute_standard_late_interaction(
            q_low.unsqueeze(0), 
            target_low.unsqueeze(0), 
            q_weights=q_w.unsqueeze(0)
        ).item()

# ========= 运行评测 =========
def run_evaluation(json_input, data_root, model_path, output_dir):
    # 1. 加载模型
    evaluator = SCMLIREvaluator(model_path)
    
    # 2. 使用工具函数加载数据
    print("Loading captions...")
    captions = load_all_captions(data_root)
    
    # 3. 初始化处理器
    img_processor = AutoImageProcessor.from_pretrained(evaluator.args.vision_model)

    with open(json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4. 汇总所有图片 ID
    unique_ids = set()
    for item in data:
        unique_ids.add(item['id'])
        unique_ids.add(item['gold'])
        for c in item['candidates']:
            unique_ids.add(c)

    # 5. 预计算所有特征
    img_feats = {} # 存储 (features, weights)
    txt_feats = {} # 存储 features
    
    print(f"Pre-calculating features for {len(unique_ids)} samples...")
    for iid in tqdm(unique_ids):
        try:
            path = get_image_path(data_root, iid)
            if os.path.exists(path):
                f, w = evaluator.extract_img_feature(path, img_processor)
                img_feats[iid] = (f, w)
                
                # 如果该 ID 对应的文本存在
                if iid in captions:
                    txt_feats[iid] = evaluator.extract_txt_feature(captions[iid])
        except Exception as e:
            print(f"Error processing {iid}: {e}")

    # 6. 计算检索分数
    i2i_results = []
    i2t_results = []

    for item in tqdm(data, desc="Ranking"):
        qid = item['id']
        if qid not in img_feats:
            continue
            
        q_low, q_w = img_feats[qid]
        candidates = [item['gold']] + item['candidates']
        
        s_ii, s_it = {}, {}
        for cid in candidates:
            # Image to Image
            if cid in img_feats:
                sim = evaluator.compute_scores(q_low, q_w, img_feats[cid][0])
                s_ii[cid] = round(sim, 4)
            else:
                s_ii[cid] = 0.0
            
            # Image to Text
            if cid in txt_feats:
                sim = evaluator.compute_scores(q_low, q_w, txt_feats[cid])
                s_it[cid] = round(sim, 4)
            else:
                s_it[cid] = 0.0

        i2i_results.append({"id": qid, "scores": s_ii})
        i2t_results.append({"id": qid, "scores": s_it})

    # 7. 保存
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scmlir_img2img_result.json"), 'w') as f:
        json.dump(i2i_results, f, indent=4)
    with open(os.path.join(output_dir, "scmlir_img2txt_result.json"), 'w') as f:
        json.dump(i2t_results, f, indent=4)
        
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    DATA_ROOT = "/home/users/h/hej/scratch/dataset/rocov2"
    JSON_INPUT = "rocov2_48x10_kis_benchmark.json"
    MODEL_PATH = "/home/users/h/hej/project/mlir4radiology/save/roco/scmlir_v2/checkpoints/scmlir_model.pth"
    OUTPUT_DIR = "prediction"

    run_evaluation(JSON_INPUT, DATA_ROOT, MODEL_PATH, OUTPUT_DIR)