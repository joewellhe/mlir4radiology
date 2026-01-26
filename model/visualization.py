import os
import sys

# Ensure we can import CMliR from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoImageProcessor
import argparse
from model.SCMliR import SCMLIR

def visualize_matplotlib(model, processor, image_paths, save_path, device='cuda', top_k=5):
    model.eval()
    
    # 1. 预处理
    pixel_values_list = []
    pil_images = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        pil_images.append(img)
        # 获取 processor 处理后的 tensor
        inputs = processor(images=img, return_tensors="pt")
        pixel_values_list.append(inputs.pixel_values.to(device))
    
    with torch.no_grad():
        # [关键修改]: 调用你模型原生的 encode_img 逻辑
        # 这样 img_embeds 才会经过 llama_proj 变成 4096 维
        img_embeds, _ = model.encode_img(pixel_values_list) 
        
        # 现在 img_embeds 是 (49, 4096)，可以正常喂给 weighter_proj 了
        weights = model.get_semantic_weights(img_embeds)
        weights = weights.squeeze(0).cpu().numpy()

    # --- 以下绘图逻辑保持不变 ---
    grid_size = int(np.sqrt(len(weights))) 
    top_indices = np.argsort(weights)[::-1][:top_k]

    for img_idx, img in enumerate(pil_images):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img)
        w, h = img.size
        patch_h, patch_w = h / grid_size, w / grid_size
        
        for rank, idx in enumerate(top_indices):
            row, col = idx // grid_size, idx % grid_size
            y_pos, x_pos = row * patch_h, col * patch_w
            
            rect = patches.Rectangle((x_pos, y_pos), patch_w, patch_h, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_pos, y_pos - 5, f"#{rank+1} ({weights[idx]:.3f})", 
                    color='red', fontsize=10, weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.axis('off')
        out_name = os.path.join(save_path, f"vis_plt_img{img_idx}.png")
        plt.savefig(out_name, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    # ... (参数解析部分与前一个脚本一致)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_dir', type=str, default='test_img1')
    parser.add_argument('--save_path', type=str, default='vis_results')
    args = parser.parse_args()

    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = SCMLIR(cfg)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    processor = AutoImageProcessor.from_pretrained(cfg.vision_model)

    image_paths = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths.sort()
    visualize_matplotlib(model, processor, image_paths, args.save_path, device)