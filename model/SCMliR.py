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
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

import pandas as pd
import tqdm
import random
import numpy as np
from PIL import Image

# 强制屏蔽非 Error 级别的日志
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


class ResidualProjector(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, mid_dim)
        self.input_ln = nn.LayerNorm(mid_dim)

        self.res_block1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 2),
            nn.GELU(),
            nn.Linear(mid_dim * 2, mid_dim),
            nn.LayerNorm(mid_dim)
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 2),
            nn.GELU(),
            nn.Linear(mid_dim * 2, mid_dim),
            nn.LayerNorm(mid_dim)
        )
        self.final_proj = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.input_ln(self.input_proj(x))
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
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
        self.rag_input_proj = nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)
        # B. 可学习的 Latent Queries (32个)
        self.num_rag_tokens = 32
        # 初始化为一个正态分布，作为"探针"去RAG里找信息
        self.rag_queries = nn.Parameter(torch.randn(1, self.num_rag_tokens, self.llama_model.config.hidden_size))
        
        # C. 使用Transformer Decoder Layer
        # 这一层包含了: Self-Attn (Query内部交互) -> Cross-Attn (Query查Text) -> FFN -> Norm
        self.vision_interactor = nn.TransformerDecoderLayer(
            d_model=self.llama_model.config.hidden_size, # 4096
            nhead=4,                                    
            dim_feedforward=self.llama_model.config.hidden_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # (Batch, Seq, Dim)
            norm_first=True    # Pre-Norm 训练更稳定
        )

        self.rag_transformer_layer = nn.TransformerDecoderLayer(
            d_model=self.llama_model.config.hidden_size, # 4096
            nhead=4,                                    
            dim_feedforward=self.llama_model.config.hidden_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # (Batch, Seq, Dim)
            norm_first=True    # Pre-Norm 训练更稳定
        )


        # 两个分布对齐层
        self.img_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.txt_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

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

            
            # =============临时 过滤掉rag_modules的参数 让其重新训练===========
            # rag_modules = ["rag_input_proj",  "rag_queries",  "rag_transformer_layer", "vision_interactor", "img_norm", "txt_norm"]
            # old_keys_count = len(state_dict)
            # state_dict = {
            #     k: v for k, v in state_dict.items() 
            #     if not any(m in k for m in rag_modules)
            # }
            # new_keys_count = len(state_dict)
            # if old_keys_count != new_keys_count:
            #     print(f"Temporary Filter: Removed {old_keys_count - new_keys_count} RAG-related parameters to force re-training.")
            # ===============================================================
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
        sim_reports: 检索到的参考文本列表
        img_embeds: (B, 49, 4096) 经过 llama_proj 的图像 Patch 特征
        """
        self.llama_tokenizer.padding_side = "right"
        ll_inputs = self.llama_tokenizer(
            sim_reports,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            t_outputs = self.embed_tokens(ll_inputs["input_ids"])  # (B, L, 4096)
        
        # 投影到 Llama 维度 -> rag_memory: (B, L, 4096)
        rag_memory = self.rag_input_proj(t_outputs)
        
        # 2. 准备初始可学习 Queries (B, 32, 4096)
        batch_size = len(sim_reports)
        base_query = self.rag_queries.expand(batch_size, -1, -1)
        
        # 3. 第一阶段：Query 读图 (Vision Interaction)
        # img_embeds 是 (B, 49, 4096)，进行分布归一化
        norm_img_embeds = self.img_norm(img_embeds)
        
        # query_with_v 包含了图像的精华信息
        query_with_v = self.vision_interactor(
            tgt=base_query, 
            memory=norm_img_embeds,
            memory_key_padding_mask=None # 图像 patch 通常无 mask
        )
        # 4. 第二阶段：带着图像信息的 Query 去提取文本 (Text Interaction)
        # 对文本 memory 进行归一化
        norm_rag_memory = self.txt_norm(rag_memory)
        key_padding_mask = (ll_inputs.attention_mask == 0)
        
        # 提取参考文本中的相关特征
        extracted_text_feat = self.rag_transformer_layer(
            tgt=query_with_v, 
            memory=norm_rag_memory, 
            memory_key_padding_mask=key_padding_mask
        )
        
        # 5. 最终融合 (加上残差，确保图像信息不丢失)
        ref_embeds = extracted_text_feat
        return ref_embeds

    def prompt_wrap(self, img_embeds, atts_img, ids=None):
        """
        调整后的 RAG Prompt 逻辑：
        结构：[Human: <Img>] + [Img] + Reference: <Ref> + [Ref_Embeds] + </Ref> + [Instruct]
        """
        p_before_list = []
        p_ref_start_list = []
        p_ref_end_list = []
        p_after_list = []
        batch_ref_embeds = None 
        
        # 定义各阶段文字占位符
        base_prompt_before = 'Human: <Img>'
        base_ref_start = 'Reference: <Ref>'
        base_ref_end = '</Ref>'
        base_prompt_after = f' {self.prompt}\nAssistant:'

        if self.args.RAG_prompt and ids is not None and hasattr(self, 'similar_cases'):
            device = img_embeds.device
            batch_size = img_embeds.shape[0]
            gate_weights = torch.ones(batch_size, 1).to(device) 
            sim_reports = []
            for i in range(len(ids)):
                curr_id = str(ids[i])
                txt, score = "", 0.0
                if curr_id in self.similar_cases:
                    txt = self.similar_cases[curr_id].get('similar_report', "")
                    score = self.similar_cases[curr_id].get('score', 0.0)
                
                # 阈值过滤与记录
                sim_reports.append(txt)
                
                # 为每个 batch 成员构建列表
                p_before_list.append(base_prompt_before)
                p_ref_start_list.append(base_ref_start)
                p_ref_end_list.append(base_ref_end)
                p_after_list.append(base_prompt_after)

                is_filtered = False
                # if score < 0.75: 
                #     is_filtered = True
                if score < 0.88: 
                    is_filtered = True
                r = random.random()
                if self.training:
                    if r < 0.1: 
                        is_filtered = True

                if is_filtered:
                    gate_weights[i] = 0.0
            # 保持 img_embeds 为 3D (B, 49, 4096) 传入，避免 Transformer 维度报错
            batch_ref_embeds = self.encode_rag_context(sim_reports, img_embeds, img_embeds.device)
                
        else:
            # 非 RAG 模式保持原样或精简
            for _ in range(img_embeds.shape[0]):
                p_before_list.append(base_prompt_before)
                p_after_list.append(base_prompt_after)

        device = img_embeds.device
        
        # 1. 编码所有文本段
        def tokenize_to_embeds(text_list):
            tokens = self.llama_tokenizer(text_list, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            return self.embed_tokens(tokens.input_ids), tokens.attention_mask

        p_before_embeds, p_before_mask = tokenize_to_embeds(p_before_list)
        p_after_embeds, p_after_mask = tokenize_to_embeds(p_after_list)

        if self.args.RAG_prompt:
            p_ref_start_embeds, p_ref_start_mask = tokenize_to_embeds(p_ref_start_list)
            p_ref_end_embeds, p_ref_end_mask = tokenize_to_embeds(p_ref_end_list)
            
            # 生成 ref embedding 的 mask
            ref_atts = torch.ones((batch_ref_embeds.shape[0], batch_ref_embeds.shape[1]), 
                                  dtype=atts_img.dtype, device=device)
            g = (gate_weights > 0).to(ref_atts.dtype)  # (B,1)
            p_ref_start_mask = p_ref_start_mask * g
            ref_atts         = ref_atts * g
            p_ref_end_mask   = p_ref_end_mask * g

            # 2. 按照新结构拼接：[Before] + [Img] + [Ref_Start] + [Ref_Embeds] + [Ref_End] + [After]
            wrapped_img_embeds = torch.cat([
                p_before_embeds, 
                img_embeds, 
                p_ref_start_embeds, 
                batch_ref_embeds, 
                p_ref_end_embeds, 
                p_after_embeds
            ], dim=1)
            
            wrapped_atts_img = torch.cat([
                p_before_mask, 
                atts_img, 
                p_ref_start_mask, 
                ref_atts, 
                p_ref_end_mask, 
                p_after_mask
            ], dim=1)

        else:
            # 基础模式：[Before] + [Img] + [After]
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = torch.cat([p_before_mask, atts_img, p_after_mask], dim=1)
            
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

    def configure_optimizers(self):
        retrieval_modules = ["img_shared_proj", "txt_shared_proj", "weighter_proj", "semantic_anchors"]
        backbone_modules = ["visual_encoder", "llama_proj", "layer_norm"]

        rag_small_modules = ["rag_input_proj", "rag_queries", "img_norm", "txt_norm"]
        rag_big_modules = ["vision_interactor", "rag_transformer_layer"]

        params_retrieval = []
        params_backbone = []
        params_rag_small = []
        params_rag_big = []

        print(f"\n[Configuring Optimizers] Mode: {'RETRIEVAL ONLY' if self.args.retrieval_only else ('RAG' if self.args.RAG_prompt else 'BACKBONE + GENERATION')}")

        for name, param in self.named_parameters():

            if self.args.retrieval_only:
                if any(r in name for r in retrieval_modules):
                    param.requires_grad = True
                    params_retrieval.append(param)
                else:
                    param.requires_grad = False

            elif self.args.RAG_prompt:
                if any(r in name for r in rag_big_modules):
                    param.requires_grad = True
                    params_rag_big.append(param)

                elif any(r in name for r in rag_small_modules):
                    param.requires_grad = True
                    params_rag_small.append(param)

                else:
                    param.requires_grad = False

            else:
                if any(b in name for b in backbone_modules):
                    param.requires_grad = True
                    params_backbone.append(param)
                else:
                    param.requires_grad = False

        # ===== Learning rates =====
        base_lr = float(self.hparams.learning_rate)  # 你现在是 5e-4（太大，只适合小模块）
        # ✅ 推荐：小模块用较大 LR，大模块用小 LR
        lr_lora = 1e-4
        lr_rag_small = min(base_lr, 2e-4)    # 把 5e-4 clamp 掉
        lr_rag_big = 1e-5                   # 先稳住；你可以后面调到 2e-5/3e-5
        lr_backbone = base_lr               # 非RAG模式下按你原策略
        lr_retrieval = base_lr

        params_list = []

        if self.args.retrieval_only:
            if params_retrieval:
                params_list.append({"params": params_retrieval, "lr": lr_retrieval, "weight_decay": 0.01})

        elif self.args.RAG_prompt:
            if params_rag_small:
                params_list.append({"params": params_rag_small, "lr": lr_rag_small, "weight_decay": 0.01})
            if params_rag_big:
                params_list.append({"params": params_rag_big, "lr": lr_rag_big, "weight_decay": 0.01})
        else:
            if params_backbone:
                params_list.append({"params": params_backbone, "lr": lr_backbone, "weight_decay": 0.01})

        optimizer = torch.optim.AdamW(params_list, betas=(0.9, 0.95), eps=1e-8)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.05)
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        print(f"LRs: base={base_lr} rag_small={lr_rag_small} rag_big={lr_rag_big} lora={lr_lora}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
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

            # mimic-cxr
            filtered_teacher = torch.where(teacher_sim > 0.93, teacher_sim, torch.tensor(-1e9).to(teacher_sim.device))

            # iu xray
            # filtered_teacher = torch.where(teacher_sim > 0.95, teacher_sim, torch.tensor(-1e9).to(teacher_sim.device))

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
            loss = 0.5 * loss_global + 0.8 * loss_main + 1.2 * loss_li + 0.5 * loss_ortho            # return {"loss": loss, "loss_main": loss_main, "pos_count": pos_mask.sum(1).mean()}
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
            # atts_bos = atts_img[:, :1]
            atts_bos = torch.ones((batch_size, 1), dtype=atts_img.dtype, device=atts_img.device)

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
            "rag_input_proj",  "rag_queries",  "rag_transformer_layer", "vision_interactor", "img_norm", "txt_norm",
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
        # atts_bos = atts_img[:, :1]
        atts_bos = torch.ones((batch_size, 1), dtype=atts_img.dtype, device=atts_img.device)

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
        # atts_bos = atts_img[:, :1]
        atts_bos = torch.ones((batch_size, 1), dtype=atts_img.dtype, device=atts_img.device)
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