import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel, AutoImageProcessor

import pandas as pd
import tqdm
import random
import numpy as np
from PIL import Image

# 强制屏蔽非 Error 级别的日志
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


# class ResidualProjector(nn.Module):
#     def __init__(self, in_dim, mid_dim, out_dim):
#         super().__init__()
#         self.input_proj = nn.Linear(in_dim, mid_dim)
#         self.input_ln = nn.LayerNorm(mid_dim)

#         self.res_block1 = nn.Sequential(
#             nn.Linear(mid_dim, mid_dim * 2),
#             nn.GELU(),
#             nn.Linear(mid_dim * 2, mid_dim),
#             nn.LayerNorm(mid_dim)
#         )

#         self.res_block2 = nn.Sequential(
#             nn.Linear(mid_dim, mid_dim * 2),
#             nn.GELU(),
#             nn.Linear(mid_dim * 2, mid_dim),
#             nn.LayerNorm(mid_dim)
#         )
#         self.final_proj = nn.Linear(mid_dim, out_dim)

#     def forward(self, x):
#         x = self.input_ln(self.input_proj(x))
#         x = x + self.res_block1(x)
#         x = x + self.res_block2(x)
#         return self.final_proj(x)

class ResidualProjector(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, num_layers=6):
        super().__init__()
        # 1. 入口层：把维度对齐到 mid_dim
        self.input_proj = nn.Linear(in_dim, mid_dim)
        self.input_ln = nn.LayerNorm(mid_dim)
        
        # 2. 深层残差块 (FFN Style)
        # 结构：Norm -> Linear(升维) -> GELU -> Dropout -> Linear(降维) -> Dropout
        # 这种 "Inverted Bottleneck" 结构是 Transformer 强大的核心原因
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mid_dim),              # Pre-Norm: 训练更稳
                nn.Linear(mid_dim, mid_dim * 4),    # 升维: 增加特征解构能力
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mid_dim * 4, mid_dim),    # 降维: 压缩信息
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # 3. 出口层
        self.final_ln = nn.LayerNorm(mid_dim)       # 出口再加个 Norm
        self.final_proj = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        # x: [Batch, Patches, in_dim]
        
        # Input Projection
        x = self.input_proj(x)
        x = self.input_ln(x) # 初始 Norm
        
        # Deep Residual Loop
        for block in self.blocks:
            # 残差连接：x + F(x)
            # 注意：因为 block 内部第一层是 Norm，所以这是标准的 Pre-Norm ResNet 结构
            x = x + block(x)
            
        # Final Projection
        x = self.final_ln(x) # 最后一层 Norm 保证输出分布稳定
        return self.final_proj(x)


class SCMLIR(pl.LightningModule):
    """
    MLIR model with Latent Retrieval RAG.
    Now using standard nn.TransformerDecoderLayer for cleaner implementation.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # ---------------------------------------------------------
        # 1. Backbone Components
        # ---------------------------------------------------------
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)

        print(f'Loading LLAMA model: {args.llama_model}')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0  
        self.llama_model = LlamaForCausalLM.from_pretrained(args.llama_model)

        # Backbone 的连接层
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

        self.embed_tokens = self.llama_model.get_input_embeddings()
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        
        # ---------------------------------------------------------
        # 2. Retrieval Components (MedCPT)
        # ---------------------------------------------------------
        # MedCPT Teacher (Always Frozen)
        print("Init Retrieval Components Param: Enabling Latent Retrieval with Standard Transformer Layer...")

        self.medcpt_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.medcpt_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        for param in self.medcpt_model.parameters():
            param.requires_grad = False
        
        self.proj_dim = 128 
        self.img_shared_proj = ResidualProjector(self.llama_model.config.hidden_size, 1024, self.proj_dim)
        self.txt_shared_proj = ResidualProjector(768, 1024, self.proj_dim)


        # Semantic Anchors
        self.num_concepts = 64 
        self.semantic_anchors = nn.Parameter(torch.randn(self.num_concepts, self.proj_dim))
        self.weighter_proj = ResidualProjector(self.llama_model.config.hidden_size, 1024, self.proj_dim)
        

        # ---------------------------------------------------------
        # 3. RAG / Prompt Components configuration
        # ---------------------------------------------------------
        print("Init RAG Prompt Param: Enabling Latent Retrieval with Standard Transformer Layer...")
        
        # RAG 转换模块
        # A. 维度对齐: MedCPT (768) -> Llama (4096)
        self.rag_input_proj = nn.Linear(768, self.llama_model.config.hidden_size)
        self.img_to_query_proj = nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)
        # B. 可学习的 Latent Queries (32个)
        self.num_rag_tokens = 32
        # 初始化为一个正态分布，作为"探针"去RAG里找信息
        self.rag_queries = nn.Parameter(torch.randn(1, self.num_rag_tokens, self.llama_model.config.hidden_size))
        
        # C. 使用Transformer Decoder Layer
        # 这一层包含了: Self-Attn (Query内部交互) -> Cross-Attn (Query查Text) -> FFN -> Norm
        self.rag_transformer_layer = nn.TransformerDecoderLayer(
            d_model=self.llama_model.config.hidden_size, # 4096
            nhead=4,                                    
            dim_feedforward=self.llama_model.config.hidden_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # (Batch, Seq, Dim)
            norm_first=True    # Pre-Norm 训练更稳定
        )
        # Misc
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if self.args.RAG_prompt:
            if hasattr(self.args, 'similar_cases_file') and self.args.similar_cases_file and os.path.exists(self.args.similar_cases_file):
                print(f"Loading similar cases from {self.args.similar_cases_file}")
                with open(self.args.similar_cases_file, 'r') as f:
                    self.similar_cases = json.load(f)

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'), weights_only=False)['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    @torch.no_grad()
    def get_image_projections(self, images):
        img_embeds, _ = self.encode_img(images)
        img_low = self.img_shared_proj(img_embeds) 
        weights = self.get_semantic_weights(img_embeds)
        return img_low, weights

    # ==========================================================
    # [NEW] Latent Retrieval Encoder (Simplified with Transformer API)
    # ==========================================================
    def encode_rag_context(self, sim_reports, img_embeds, device):
        """
        sim_reports: 文本列表
        img_embeds: (B, 4096) 这里的 img_embeds 已经是经过 llama_proj 映射过的图片特征
        """
        # 1. MedCPT 编码文本 (Memory)
        t_inputs = self.medcpt_tokenizer(sim_reports, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            t_outputs = self.medcpt_model(**t_inputs).last_hidden_state # (B, L, 768)
        
        # 投影到 Llama 维度 -> Memory (B, L, 4096)
        rag_memory = self.rag_input_proj(t_outputs)
        
        # 2. 准备 Latent Queries (Target)
        # 原始的 learnable query: (1, 32, 4096)
        batch_size = len(sim_reports)
        base_query = self.rag_queries.expand(batch_size, -1, -1)
        
        # [CRITICAL] 注入图像信息！
        # img_embeds: (B, 4096) -> unsqueeze -> (B, 1, 4096)
        img_guidance = self.img_to_query_proj(img_embeds).unsqueeze(1)
        
        # 将图像加到 Query 上 (Residual 方式)
        # 现在的 Query = "我想找信息的意图(32个槽位)" + "这张图的语义特征"
        # 广播机制: (B, 32, 4096) + (B, 1, 4096)
        tgt_query = base_query + img_guidance
        
        # 3. Mask
        key_padding_mask = (t_inputs.attention_mask == 0)

        # 4. Transformer Interaction
        ref_embeds = self.rag_transformer_layer(
            tgt=tgt_query, 
            memory=rag_memory, 
            memory_key_padding_mask=key_padding_mask
        )
        
        return ref_embeds

    def prompt_wrap(self, img_embeds, atts_img, ids=None):
        """
        RAG Prompt 核心逻辑：Latent Embeddings 拼在前面
        """
        p_before_list = []
        p_after_list = []
        batch_ref_embeds = None 
        
        base_prompt_before = 'Human: <Img>'
        base_prompt_after = f'</Img> {self.prompt}\nAssistant:'

        if self.args.RAG_prompt and ids is not None and hasattr(self, 'similar_cases'):
            sim_reports = []
            
            for i in range(len(ids)):
                curr_id = str(ids[i])
                txt = ""
                score = 0.0
                if curr_id in self.similar_cases:
                    txt = self.similar_cases[curr_id].get('similar_report', "")
                    score = self.similar_cases[curr_id].get('score', 0.0)
                
                # 阈值过滤
                if score < 0.9: txt = ""
                # Dropout
                if self.training and random.random() < 0.4: txt = ""
                
                sim_reports.append(txt)
                p_before_list.append(base_prompt_before)
                p_after_list.append(base_prompt_after)

            processed_reports = [s if s != "" else "empty" for s in sim_reports]
            # [FIX] 传入 img_embeds
            # 注意 img_embeds 此时 shape 是 (B, 49, 4096)
            # 建议传 mean pooling 后的: img_embeds.mean(dim=1) 如果是序列的话
            img_global = img_embeds.mean(dim=1)
            batch_ref_embeds = self.encode_rag_context(processed_reports, img_global, img_embeds.device)
                
        else:
            for _ in range(img_embeds.shape[0]):
                p_before_list.append(base_prompt_before)
                p_after_list.append(base_prompt_after)

        device = img_embeds.device
        
        p_before_tokens = self.llama_tokenizer(p_before_list, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        p_after_tokens = self.llama_tokenizer(p_after_list, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)
        
        if self.args.RAG_prompt:
            # 有 RAG: [Ref(32)] + [Human: <Img>] + [Img] + [Instruct]
            ref_atts = torch.ones((batch_ref_embeds.shape[0], batch_ref_embeds.shape[1]), dtype=atts_img.dtype, device=device)
            
            wrapped_img_embeds = torch.cat([batch_ref_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = torch.cat([ref_atts, p_before_tokens.attention_mask, atts_img, p_after_tokens.attention_mask], dim=1)
        else:
            # 无 RAG
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = torch.cat([p_before_tokens.attention_mask, atts_img, p_after_tokens.attention_mask], dim=1)
            
        return wrapped_img_embeds, wrapped_atts_img

    # ==========================================================
    # Semantic Weight & Late Interaction (Unchanged)
    # ==========================================================
    def get_semantic_weights(self, img_embeds):
        img_feats = F.normalize(self.weighter_proj(img_embeds), dim=-1)
        anchors = F.normalize(self.semantic_anchors, dim=-1) 
        sim_to_anchors = torch.matmul(img_feats, anchors.t())
        importance_score = sim_to_anchors.max(dim=-1).values
        weights = F.softmax(importance_score / 0.2, dim=-1) 
        return weights

    def compute_standard_late_interaction(self, q_tokens, t_tokens, q_weights=None, temperature=0.05):
        q = F.normalize(q_tokens, dim=-1)
        t = F.normalize(t_tokens, dim=-1)
        sim_matrix = torch.matmul(q.unsqueeze(1), t.unsqueeze(0).transpose(-1, -2))
        
        if self.training:
            max_sim = temperature * torch.logsumexp(sim_matrix / temperature, dim=-1)
        else:
            max_sim = sim_matrix.max(dim=-1).values
            
        # max_sim = F.softplus(max_sim)
        
        if q_weights is not None:
            w = q_weights.unsqueeze(1).expand(-1, q_tokens.size(0), -1)
            scores = (max_sim * w).sum(dim=-1)
            w_sum = w.sum(dim=-1).clamp(min=1e-9)
            scores = scores / w_sum
        else:
            scores = max_sim.mean(dim=-1)
        return scores
    
        # if q_weights is not None:
        #     # 2. 移除归一化除法 (scores / w_sum)
        #     # 直接计算加权证据的累加和。w 代表图像中不同区域（token）的重要性。
        #     scores = (max_sim * q_weights).sum(dim=-1) 
        # else:
        #     # 3. 同样将平均改为累加
        #     scores = max_sim.sum(dim=-1)

        # return scores

    def soft_info_nce_loss(self, student_logits, teacher_sim, distill_temp=4.0):
        teacher_probs = teacher_sim / (teacher_sim.sum(dim=1, keepdim=True) + 1e-9)
        student_log_probs = F.log_softmax(student_logits / distill_temp, dim=1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return loss

    def multi_label_bce_loss(self, student_sim, teacher_sim):
        targets = teacher_sim.clamp(0, 1) 
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn(student_sim, targets)
        return loss

    # ==========================================================
    # Optimizers (Updated for new Layer)
    # ==========================================================
    def configure_optimizers(self):
        retrieval_modules = ["img_shared_proj", "txt_shared_proj", "weighter_proj", "semantic_anchors"]
        backbone_modules = ["visual_encoder", "llama_proj", "layer_norm"]
        rag_modules = ["rag_input_proj",  "rag_queries",  "rag_transformer_layer", "img_to_query_proj"]

        params_to_update = []
        print(f"\n[Configuring Optimizers] Mode: {'RETRIEVAL ONLY' if self.args.retrieval_only else 'BACKBONE + GENERATION'}")
        
        for name, param in self.named_parameters():
            if self.args.retrieval_only:
                if any(r in name for r in retrieval_modules):
                    param.requires_grad = True
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
            
            elif self.args.RAG_prompt:
                # 只训练新加的 RAG Transformer 结构
                if any(r in name for r in rag_modules):
                    param.requires_grad = True
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
            else:
                if any(b in name for b in backbone_modules):
                    param.requires_grad = True
                    params_to_update.append(param)
                else:
                    param.requires_grad = False

        print(f"Total trainable params: {len(params_to_update)}")
        
        lr = self.hparams.learning_rate
        if self.args.RAG_prompt:
             # Transformer LR
             lr = 1e-4 

        optimizer = torch.optim.AdamW(params_to_update, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        scheduler = None
        if self.args.dataset == 'mimic_cxr':
            # 2. 计算总步数 (Total Steps) 和 Warmup 步数
            total_steps = self.trainer.estimated_stepping_batches
            
            # 设定 Warmup 为总步数的 5% 到 10%
            warmup_steps = int(total_steps * 0.05) 
            
            print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

            # 3. 使用 HuggingFace 的标准 Warmup + Cosine Decay 调度器
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            # 4. 返回配置字典
            # 注意：Warmup 调度器必须按 'step' (每个 batch) 更新，而不是按 'epoch' 更新
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ==========================================================
    # Forward & Steps
    # ==========================================================
    def forward(self, samples):
        # ==========================================================
        # Branch A-1: ROCO Retrieval Only (Hard Negative Batch)
        # ==========================================================
        if self.args.retrieval_only and getattr(self.args, 'dataset', None) == 'roco':

            image = samples["image"]
            # b, n, c, h, w = samples["image"].shape
            # print(f"Retrieval-Only Mode: Processing batch with shape {samples['image'].shape}")
            img_embeds, atts_img = self.encode_img(image)
            raw_texts = samples["input_text"]
            all_texts = [t[0] for t in raw_texts]

            # 由于你在 DataLoader 里已经扩展了图像，这里的 image 应该是 (16, 3, H, W)
            # img_embeds 经过 encode_img 后应为 (16, 49, 4096)
            batch_size = len(all_texts) # 应该是 16
            
            # 2. MedCPT Teacher 编码
            self.medcpt_model.eval()
            with torch.no_grad():
                t_inputs = self.medcpt_tokenizer(all_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(img_embeds.device)
                t_outputs = self.medcpt_model(**t_inputs)
                t_mask = t_inputs.attention_mask
                t_seq_768 = F.normalize(t_outputs.last_hidden_state, dim=-1)
                t_global = F.normalize((t_outputs.last_hidden_state * t_mask.unsqueeze(-1)).sum(1) / t_mask.sum(1, keepdim=True).clamp(min=1e-9), dim=-1)
                
                # 计算文本间的相似度作为 Label (16x16)
                teacher_sim = t_global @ t_global.t()
                # 过滤并归一化为概率分布
                filtered_teacher = torch.where(teacher_sim > 0.95, teacher_sim, torch.tensor(-1e9).to(teacher_sim.device))
                teacher_probs = F.softmax(filtered_teacher / 0.05, dim=-1)

            # 3. 学生网络特征提取与投影
            img_tok_low = F.normalize(self.img_shared_proj(img_embeds), dim=-1) # (16, 49, 128)
            t_seq_low = F.normalize(self.txt_shared_proj(t_seq_768), dim=-1)    # (16, seq, 128)
            w_i2t = self.get_semantic_weights(img_embeds)                      # (16, 49)
            
            # 4. 计算三种核心相似度矩阵
            # A. 全局特征相似度 (16x16)
            img_global = img_tok_low.mean(dim=1) # (16, 128)
            txt_global = t_seq_low.mean(dim=1)   # (16, 128)
            sim_global = torch.matmul(img_global, txt_global.t())

            # B. Late Interaction 相似度 (16x16)
            # 使用原有的 compute_standard_late_interaction，因为它现在是 B-vs-B 结构
            sim_i2t = self.compute_standard_late_interaction(img_tok_low, t_seq_low, q_weights=w_i2t, temperature=0.05)

            # C. 图像间相似度 (16x16) -> loss_li
            # 权重增强
            w_i2i = torch.pow(w_i2t, 1.2)
            w_i2i = w_i2i / (w_i2i.sum(dim=-1, keepdim=True) + 1e-9)
            sim_img2img = self.compute_standard_late_interaction(img_tok_low, img_tok_low, q_weights=w_i2i, temperature=0.05)

            # 5. 计算损失
            T = 0.05
            loss_global = F.cross_entropy(sim_global / T, teacher_probs)
            
            # 对称 Cross Entropy (I2T 和 T2I)
            loss_i2t = F.cross_entropy(sim_i2t / T, teacher_probs)
            loss_t2i = F.cross_entropy(sim_i2t.t() / T, teacher_probs.t())
            loss_main = (loss_i2t + loss_t2i) / 2
            
            # 图像自对齐损失 (在这个特殊 Batch 中，所有图其实是一样的，
            # 理论上 sim_img2img 应该接近全 1 矩阵，或者遵循 teacher_probs 的文本逻辑引导)
            loss_li = F.cross_entropy(sim_img2img / T, teacher_probs)

            # 语义锚点正交损失
            anchors = F.normalize(self.semantic_anchors, dim=-1)
            gram_matrix = torch.matmul(anchors, anchors.t())
            loss_ortho = F.mse_loss(gram_matrix, torch.eye(self.num_concepts, device=gram_matrix.device))

            # 总 Loss 加权
            # loss = 0.5 * loss_global + 0.7 * loss_main + 1.2 * loss_li + 0.5 * loss_ortho
            # loss = 0.5 * loss_global + 1 * loss_main + 1.2 * loss_li + 0.5 * loss_ortho
            loss = 1.0 * loss_global + 1.0 * loss_main + 0.5 * loss_li + 0.5 * loss_ortho

            return {
                "loss": loss, 
                "loss_main": loss_main, 
                "loss_li": loss_li,
                "loss_ortho": loss_ortho,
                "teacher_probs": teacher_probs[0, :],
                "sim_global": F.softmax(sim_global/T, dim=-1)[0, :],
                "sim_img2text": F.softmax(sim_i2t/T, dim=-1)[0, :],
                "sim_img2img": F.softmax(sim_img2img/T, dim=-1)[0, :],
                "w_i2t": w_i2t[0, :20] if 'w_i2t' in locals() else None,
            }

        # ==========================================================
        # Branch A-2: Original Retrieval (Iuxray or mimic-cxr)
        # ==========================================================

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        loss = torch.tensor(0.0, device=img_embeds.device)
        # Branch A: Retrieval Training
        if self.args.retrieval_only and "input_text" in samples:
            batch_size = img_embeds.shape[0]
            w_i2t = self.get_semantic_weights(img_embeds)
            w_i2i = torch.pow(w_i2t, 1.2)
            w_i2i = w_i2i / (w_i2i.sum(dim=-1, keepdim=True) + 1e-9)
            img_tok_low = F.normalize(self.img_shared_proj(img_embeds), dim=-1)
            
            with torch.no_grad():
                self.medcpt_model.eval()
                t_inputs = self.medcpt_tokenizer(samples["input_text"], padding=True, truncation=True, max_length=256, return_tensors="pt").to(img_embeds.device)
                t_outputs = self.medcpt_model(**t_inputs)
                t_mask = t_inputs.attention_mask
                t_seq_768 = F.normalize(t_outputs.last_hidden_state, dim=-1)
                t_global = F.normalize((t_outputs.last_hidden_state * t_mask.unsqueeze(-1)).sum(1) / t_mask.sum(1, keepdim=True).clamp(min=1e-9), dim=-1)
            
            teacher_sim = t_global @ t_global.t()
            t_seq_low = F.normalize(self.txt_shared_proj(t_seq_768), dim=-1)
            
            # pos_mask = (teacher_sim > 0.94).float() 
            # pos_mask.fill_diagonal_(1.0)
            # targets = pos_mask / (pos_mask.sum(dim=1, keepdim=True) + 1e-9)
            filtered_teacher = torch.where(teacher_sim > 0.965, teacher_sim, torch.tensor(-1e9).to(teacher_sim.device))
            teacher_probs = F.softmax(filtered_teacher / 0.05, dim=-1)
            # scale = self.logit_scale.exp().clamp(max=100)
            sim_i2t = self.compute_standard_late_interaction(img_tok_low, t_seq_low, q_weights=w_i2t, temperature=0.05)
            sim_img2img = self.compute_standard_late_interaction(img_tok_low, img_tok_low, q_weights=w_i2i, temperature=0.05)

            # 在 forward 的 Branch A 中加入
            img_global = img_tok_low.mean(dim=1) # (B, 128)
            txt_global = t_seq_low.mean(dim=1) # (B, 128)

            # 2. 计算全局相似度矩阵
            sim_global = torch.matmul(img_global, txt_global.t()) # (B, B)

            # 3. 计算全局 Loss (同样建议加温度系数 T)
            T = 0.05
            loss_global = F.cross_entropy(sim_global / T, teacher_probs)

            loss_i2t = F.cross_entropy(sim_i2t/0.05, teacher_probs)
            loss_t2i = F.cross_entropy(sim_i2t.t()/0.04, F.softmax(filtered_teacher.t() / 0.05, dim=-1))
            loss_main = (loss_i2t + loss_t2i) / 2
            # loss_li = self.multi_label_bce_loss(sim_img2img, pos_mask)
            loss_li = F.cross_entropy(sim_img2img/0.05, teacher_probs)

            anchors = F.normalize(self.semantic_anchors, dim=-1)
            gram_matrix = torch.matmul(anchors, anchors.t())
            identity = torch.eye(self.num_concepts, device=gram_matrix.device)
            loss_ortho = F.mse_loss(gram_matrix, identity)
            # loss = loss_main + loss_li + 0.5 * loss_ortho
            loss = 0.7 * loss_global + 1 * loss_main + 1 * loss_li + 0.5 * loss_ortho            # return {"loss": loss, "loss_main": loss_main, "pos_count": pos_mask.sum(1).mean()}
            return {"loss": loss, "loss_main": loss_main, "loss_li": loss_li,
                    "teacher_probs": teacher_probs[0, :] if 'teacher_sim' in locals() else None,
                    "sim_global": F.softmax(sim_global/0.05, dim=-1)[0, :],
                    "sim_img2text": F.softmax(sim_i2t/0.04, dim=-1)[0, :] if 'sim_i2t' in locals() else None,
                    "sim_img2img": F.softmax(sim_img2img/0.05, dim=-1)[0, :] if 'sim_img2img' in locals() else None,      
                    "w_i2t": w_i2t[0, :20] if 'w_i2t' in locals() else None,}

        # Branch B: Generation Training
        elif self.training and not self.args.retrieval_only:
            img_embeds = self.layer_norm(img_embeds)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, ids=samples.get("id"))
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["input_text"]]
            to_regress_tokens = self.llama_tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True,
                max_length=self.hparams.max_length, add_special_tokens=False
            ).to(image[0].device)
            
            targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == 0, -100)
            empty_targets = torch.ones([atts_img.shape[0], atts_img.shape[1] + 1], dtype=torch.long, device=image[0].device).fill_(-100)
            targets = torch.cat([empty_targets, targets], dim=1)
            
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1], dtype=to_regress_tokens.input_ids.dtype, device=image[0].device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            atts_bos = atts_img[:, :1]
            
            to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
            
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                return_dict=True, labels=targets,
            )
            return {"loss": outputs.loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        scalars_to_log = {} 
        vectors_to_print = {}
        # 标量 -> 进度条
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    scalars_to_log[k] = v  
                else:
                    vectors_to_print[k] = v.detach().cpu().tolist()
            elif isinstance(v, (float, int)):
                scalars_to_log[k] = v
        self.log_dict(scalars_to_log, prog_bar=True, sync_dist=True)

        # 3. 将向量强制打印到屏幕 (Debug)
        if self.args.retrieval_only and batch_idx % 200 == 0 and self.trainer.is_global_zero:
            # 格式化一下打印内容，保留4位小数，方便阅读
            print_msg = {k: ([round(x, 4) for x in v] if isinstance(v, list) else v) 
                        for k, v in vectors_to_print.items()}
            print(f"\n[Step {batch_idx}] DEBUG: {print_msg}", flush=True)
        return result["loss"]
        
    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        state_dict = self.state_dict()
        
        # 定义需要保存的模块
        modules_to_save = [
            "visual_encoder", "llama_proj", "layer_norm",
            "img_shared_proj", "txt_shared_proj", "weighter_proj", "semantic_anchors",
            "rag_input_proj", "rag_queries", "rag_transformer_layer", "img_to_query_proj"
        ]

        # 过滤 state_dict
        keys_to_keep = [k for k in state_dict.keys() if any(m in k for m in modules_to_save)]
        final_state_dict = {k: state_dict[k] for k in keys_to_keep}

        # 准备保存对象
        save_obj = {
            "model": final_state_dict, 
            "config": self.hparams, 
            "epoch": current_epoch, 
            "step": global_step
        }
        
        checkpoint_dir = os.path.join(self.hparams.savedmodel_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_save_path = os.path.join(checkpoint_dir, "scmlir_model.pth")
        torch.save(save_obj, model_save_path)
        
        # 2. checkpoint_info.txt
        # 使用追加模式 'a' 可以保留历史记录，或者使用 'w' 只保留最新的
        info_save_path = os.path.join(checkpoint_dir, "checkpoint_info.txt")
        with open(info_save_path, "a", encoding="utf-8") as f:
            stage = ""
            if not self.args.retrieval_only and not self.args.RAG_prompt:
                stage ="backbone train" 
            elif self.args.retrieval_only:
                stage = "retrieval train"
            elif self.args.RAG_prompt:
                stage = "RAG train"
            f.write(f"{stage}: epoch {current_epoch} global step {global_step} valid_results {eval_res}\n")

        self.print(f"Model saved to {model_save_path}")
        self.print(f"Metrics logged to {info_save_path}")
        
    def validation_step(self, samples, batch_idx):
        if self.args.retrieval_only: 
            outputs = self.forward(samples)
        
            # 提取各个部分的 Loss
            val_losses = {
                "val_loss": outputs["loss"].detach(),
                "val_loss_main": outputs["loss_main"].detach(),
                "val_loss_li": outputs["loss_li"].detach()
            }
            self.val_step_outputs.append(val_losses)
            return val_losses
        
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'], 
            return_tensors="pt", 
            padding="longest", 
            truncation=False,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        
        # Latent RAG
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, ids=samples.get("id"))

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=atts_img.dtype, device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            num_beams=self.hparams.beam_size, do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens, max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty, length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )

        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text.replace('\n', ' ')
        return output_text

    def on_train_epoch_end(self):
        if self.trainer.limit_val_batches == 0.0:
            current_epoch = self.trainer.current_epoch
            
            # 每 10 个 epoch 保存一次
            if self.trainer.is_global_zero:
                self.print(f"\n[Epoch {current_epoch}] Skipping validation, saving checkpoint by epoch interval...")
                fake_metrics = {"metrics": "n"}
                self.save_checkpoint(fake_metrics)

    def on_validation_epoch_end(self):
        if self.args.retrieval_only:
            avg_loss = torch.stack([x['val_loss'] for x in self.val_step_outputs]).mean()
            avg_loss_main = torch.stack([x['val_loss_main'] for x in self.val_step_outputs]).mean()
            avg_loss_li = torch.stack([x['val_loss_li'] for x in self.val_step_outputs]).mean()


            avg_loss_main = avg_loss_main.item()
            avg_loss_li = avg_loss_li.item()
            metrics = {
                "val_loss_main": avg_loss_main,
                "val_loss_li": avg_loss_li
            }
            
            self.log_dict(metrics, prog_bar=True, logger=True, sync_dist=True)
            self.print(f"\n[Epoch {self.trainer.current_epoch}] Validation Retrieval Loss: {avg_loss:.4f} (Main: {avg_loss_main:.4f}, LI: {avg_loss_li:.4f})")

            # just focus loss_main and loss_li
            focused_loss = 1 * avg_loss_main + 1 * avg_loss_li

            # 3. 按照 Loss 减小来保存模型 (初始化 self.best_val_loss = float('inf'))
            if self.trainer.is_global_zero:
                current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
                if current_epoch % 2 == 0:
                    self.print(f"epoch: {current_epoch} - val_loss: {focused_loss:.4f}, saving checkpoint...\n")
                    self.save_checkpoint(metrics)
            self.val_step_outputs.clear()
            return
            
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=False, logger=True, prog_bar=True, rank_zero_only=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        # json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        # json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight
        
        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'], 
            return_tensors="pt", 
            padding="longest", 
            truncation=False,
            # truncation=True, 
            # max_length=self.hparams.max_length, 
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, ids=samples.get("id"))

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=atts_img.dtype, device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            num_beams=self.hparams.beam_size, do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens, max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty, length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        if self.trainer.is_global_zero:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            
            df_res = pd.DataFrame({'ID': ids, 'Caption': hypo}).drop_duplicates(subset=['ID'])
            df_ref = pd.DataFrame({'ID': ids, 'Caption': ref}).drop_duplicates(subset=['ID'])
            df_res.to_csv(os.path.join(result_folder, 'test_result.csv'), index=False)
            df_ref.to_csv(os.path.join(result_folder, 'test_refs.csv'), index=False)

            ref = {k: [v] for k, v in zip(df_ref['ID'], df_ref['Caption'])}
            hypo = {k: [v] for k, v in zip(df_res['ID'], df_res['Caption'])}
            eval_res = self.score(ref=ref,hypo=hypo)

            json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
            json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
            self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        self.test_step_outputs.clear()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()