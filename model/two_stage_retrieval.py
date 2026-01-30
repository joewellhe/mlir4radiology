import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import faiss
import argparse
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
from tqdm import tqdm
import glob

# Ensure we can import CMliR from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [MODIFIED] Update Import to use the new Class Name
from model.SCMliR import SCMLIR

class TwoStageRetriever:
    """
    Implements Two-Stage Retrieval aligned with CMLIR architecture:
    Stage 1: Global Retrieval using FAISS (Semantic-Weighted Global Vectors)
    Stage 2: Local Reranking using Semantic-Anchored Late Interaction
    """
    def __init__(self, args):
        self.device = 'cuda'
        self.args = args
        
        print(args)
        print(f"Loading CMLIR model from {args.checkpoint}...")
        
        checkpoint = torch.load(
            args.checkpoint, 
            map_location=self.device, 
            weights_only=False
        )
        
        # 2. 参数融合与覆盖
        saved_config = checkpoint["config"]

        # 2. 定义你想确认的模块前缀
        state_dict = checkpoint['model']
        modules_to_check = [
            "visual_encoder", "llama_proj", "layer_norm",
            "img_shared_proj", "txt_shared_proj", "weighter_proj", "semantic_anchors",
            "rag_input_proj", "rag_queries", "rag_transformer_layer", "img_to_query_proj"
        ]

        print(f"{'Module Name':<25} | {'Status':<10} | {'Keys Count'}")
        print("-" * 50)

        for module in modules_to_check:
            # 查找以该模块名开头的 key
            relevant_keys = [k for k in state_dict.keys() if k.startswith(module)]
            status = "✅ 存在" if len(relevant_keys) > 0 else "❌ 缺失"
            print(f"{module:<25} | {status:<10} | {len(relevant_keys)}")

        # 额外看一眼具体名字，防止命名偏移
        print("\n前 5 个 Key 样例:")
        print(list(state_dict.keys())[:5])
        if hasattr(saved_config, "__dict__"):
            saved_dict = vars(saved_config)
        else:
            saved_dict = saved_config

        saved_config.update(vars(args)) 
        final_cfg = argparse.Namespace(**saved_config)
        final_cfg.delta_file = None 

        print("============final_cfg==============")
        print(final_cfg)

        self.model = SCMLIR(final_cfg)
        msg = self.model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Restored model weights. Missing keys: {len(msg.missing_keys)}")

        self.model.to(self.device)
        self.model.eval()
        
        # 优先使用覆盖后的 vision_model 参数
        vision_model_path = final_cfg.vision_model
        print(f"Loading AutoImageProcessor for: {vision_model_path}")
        self.processor = AutoImageProcessor.from_pretrained(vision_model_path)
        
        self.index = None
        self.local_features = []
        self.metadata = []

    def preprocess(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        tensor_list = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt")
                tensor_list.append(inputs.pixel_values.to(self.device))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if not tensor_list:
            return None
        return tensor_list

    @torch.no_grad()
    def get_features(self, image_tensors):
        """
        Extracts features using the Updated CMLIR Logic:
        1. Local Vectors (L, 128): Projected features using 'img_shared_proj'.
        2. Weights (L): Semantic weights from 'get_semantic_weights'.
        3. Global Vector (1, 128): Weighted Mean of Local Vectors.
        """
        # img_low: (B, L, 128)
        # weights: (B, L)
        img_low, weights = self.model.get_image_projections(image_tensors)
        weights = torch.pow(weights, 1.2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        # 1. Normalize Local Features
        img_proj = F.normalize(img_low, dim=-1) # (B, L, 128)

        # 2. Compute Global Feature: Weighted Mean Pooling
        w_expanded = weights.unsqueeze(-1)
        
        # Weighted Sum: (B, L, 128) * (B, L, 1) -> sum dim 1 -> (B, 128)
        global_vec = (img_proj * w_expanded).sum(dim=1)
        
        # Normalize Global Vector for FAISS (Cosine Similarity)
        global_vec = F.normalize(global_vec, p=2, dim=1)
        
        return global_vec, img_proj, weights

    def build_index(self, data_list, save_path):
        """
        Builds the retrieval index.
        """
        print(f"Building index for {len(data_list)} items...")
        index_file = os.path.join(save_path, "index.faiss")
        meta_file = os.path.join(save_path, "index_meta.pt")
        global_vecs_list = []
        local_data_list = []
        valid_metadata = []
        
        for item in tqdm(data_list, desc="Indexing"):
            img_paths = item['image_path'] # List of paths
            report = item.get('report', '') 
            item_id = item.get('id', 'unknown') # [NEW] Store ID

            img_tensors = self.preprocess(img_paths)
            if img_tensors is None: continue
            
            # g_vec: (B, 128), l_vec: (B, L, 128), w: (B, L)
            g_vec, l_vec, w = self.get_features(img_tensors)
            
            # Collect Global (CPU numpy)
            global_vecs_list.append(g_vec.cpu().numpy())
            
            # Collect Local (save as half precision to save RAM)
            local_data_list.append({
                "local": l_vec.squeeze(0).half().cpu(),   # (L, 128)
                "weights": w.squeeze(0).half().cpu()      # (L)
            })
            valid_metadata.append({'path': img_paths, 'report': report, 'id': item_id})

        if len(global_vecs_list) == 0:
            print("No valid images found to index.")
            return

        # Create FAISS Index
        global_matrix = np.vstack(global_vecs_list).astype('float32')
        dim = global_matrix.shape[1] # Should be 128
        print(f"Global Vector Dimension: {dim}")
        
        # Inner Product on Normalized Vectors = Cosine Similarity
        index = faiss.IndexFlatIP(dim)
        index.add(global_matrix)
        
        # Save everything
        faiss.write_index(index, index_file)
        torch.save({
            "local_data": local_data_list,
            "metadata": valid_metadata
        }, meta_file)
        
        self.index = index
        self.local_features = local_data_list
        self.metadata = valid_metadata
        print(f"Index built successfully. Saved to {index_file} and {meta_file}")

    def load_index(self, index_path):
        index_file = os.path.join(index_path, "index.faiss")
        meta_file = os.path.join(index_path, "index_meta.pt")
        
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            print(f"Index files not found in {index_path}")
            return

        print(f"Loading index from {index_file}...")
        self.index = faiss.read_index(index_file)
        
        print(f"Loading metadata from {meta_file}...")
        meta = torch.load(meta_file)
        self.local_features = meta["local_data"]
        self.metadata = meta["metadata"]
        print(f"Index loaded with {self.index.ntotal} items.")

    # =========================================================================
    # Generate Similar Case Dataset (Internal Optimized Search - Train on Train)
    # =========================================================================
    def create_similar_dataset(self, save_path, valid_data_list=None, top_k_global=1000, top_k_rerank=5):
        """
        Iterates over the loaded index (Train), searches for the nearest neighbor 
        (excluding self) for each item, and saves a mapping JSON.
        [MODIFIED] Also processes validation data if provided, searching in the Train index.
        """
        if self.index is None or not self.local_features:
            print("Index not loaded. Build or load index first.")
            return

        print(f"Creating similar case dataset for {self.index.ntotal} items (Train)...")
        result_mapping = {}
        
        for i in tqdm(range(self.index.ntotal), desc="Generating Train Prompts"):
            meta_item = self.metadata[i]
            current_id = meta_item.get('id', f"unknown_id_{i}") 
            
            # 1. Reconstruct Query Global Vector from FAISS (Fast)
            try:
                q_global = self.index.reconstruct(i).reshape(1, -1)
            except Exception as e:
                print(f"Error reconstructing vector {i}: {e}. Skipping.")
                continue

            # 2. Get Query Local Data from RAM
            q_local = self.local_features[i]["local"].to(self.device).float().unsqueeze(0) 
            q_weights = self.local_features[i]["weights"].to(self.device).float().unsqueeze(0)
            
            # --- Stage 1: Global Search ---
            D, I = self.index.search(q_global, top_k_global + 1)
            candidate_indices = I[0]
            
            # --- Stage 2: Local Rerank ---
            cand_locals = []
            valid_indices = []
            
            for idx in candidate_indices:
                if idx == -1: continue
                if idx == i: continue # [CRITICAL] Skip self
                
                data = self.local_features[idx]
                cand_locals.append(data["local"].to(self.device).float())
                valid_indices.append(idx)
            
            if not cand_locals: continue

            # Reranking Logic
            t_tensor = torch.stack(cand_locals)
            
            q_norm = F.normalize(q_local, dim=-1)
            t_norm = F.normalize(t_tensor, dim=-1)
            
            q_expand = q_norm.expand(len(cand_locals), -1, -1)
            sim_matrix = torch.bmm(q_expand, t_norm.transpose(1, 2))
            
            max_sim = sim_matrix.max(dim=-1).values
            w_expand = q_weights.expand(len(cand_locals), -1)
            scores = (max_sim * w_expand).sum(dim=-1)
            w_sum = w_expand.sum(dim=-1).clamp(min=1e-9)
            scores = scores / w_sum
            
            scores = scores.detach().cpu().numpy()
            best_arg = np.argmax(scores)
            
            best_idx = valid_indices[best_arg]
            best_score = scores[best_arg]
            best_meta = self.metadata[best_idx]
            
            result_mapping[current_id] = {
                "report": meta_item.get('report', ''),
                "similar_id": best_meta['id'],
                "similar_report": best_meta['report'],
                "score": float(best_score)
            }
            
        # 2. Valid on Train (New)
        if valid_data_list:
            print(f"Creating similar case dataset for {len(valid_data_list)} VALIDATION items...")
            for item in tqdm(valid_data_list, desc="Generating Valid Prompts"):
                current_id = item.get('id', 'unknown')
                img_paths = item.get('image_path', [])
                
                # Extract Features
                img_tensors = self.preprocess(img_paths)
                if img_tensors is None: continue
                
                q_global, q_local, q_weights = self.get_features(img_tensors)

                # --- Stage 1: Global Search ---
                D, I = self.index.search(q_global.cpu().numpy().astype('float32'), top_k_global)
                candidate_indices = I[0]

                # --- Stage 2: Local Rerank ---
                cand_locals = []
                valid_indices = []

                for idx in candidate_indices:
                    if idx == -1: continue
                    data = self.local_features[idx]
                    cand_locals.append(data["local"].to(self.device).float())
                    valid_indices.append(idx)
                
                if not cand_locals: continue

                t_tensor = torch.stack(cand_locals)
                q_norm = F.normalize(q_local, dim=-1)
                t_norm = F.normalize(t_tensor, dim=-1)
                
                q_expand = q_norm.expand(len(cand_locals), -1, -1)
                sim_matrix = torch.bmm(q_expand, t_norm.transpose(1, 2))
                
                max_sim = sim_matrix.max(dim=-1).values
                w_expand = q_weights.expand(len(cand_locals), -1)
                scores = (max_sim * w_expand).sum(dim=-1)
                w_sum = w_expand.sum(dim=-1).clamp(min=1e-9)
                scores = scores / w_sum
                
                scores = scores.detach().cpu().numpy()
                best_arg = np.argmax(scores)
                
                best_idx = valid_indices[best_arg]
                best_score = scores[best_arg]
                best_meta = self.metadata[best_idx]

                result_mapping[current_id] = {
                    "report": item.get('report', ''),
                    "similar_id": best_meta['id'],
                    "similar_report": best_meta['report'],
                    "score": float(best_score)
                }

        json_path = os.path.join(save_path, "similar_cases.json")
        with open(json_path, 'w') as f:
            json.dump(result_mapping, f, indent=2)
        print(f"Saved similar case mapping to {json_path}")

    # =========================================================================
    # [NEW] Generate Test Similar Cases (Search Test on Train Index)
    # =========================================================================
    def create_test_similar_dataset(self, test_data_list, save_path, top_k_global=1000, top_k_rerank=10):
        """
        Reads external Test List, extracts features, searches in loaded Index (Train),
        and saves mapping JSON.
        """
        if self.index is None:
            print("Index not loaded. Please load the TRAINING index first.")
            return

        print(f"Creating similar case dataset for {len(test_data_list)} TEST items...")
        result_mapping = {}

        for item in tqdm(test_data_list, desc="Generating Test Prompts"):
            current_id = item.get('id', 'unknown')
            img_paths = item.get('image_path', [])
            
            # 1. Extract Features from Test Image (Query)
            # This is slow because we run the encoder, but necessary for new data
            img_tensors = self.preprocess(img_paths)
            if img_tensors is None: continue
            
            # q_global: (1, 128), q_local: (1, L, 128), q_weights: (1, L)
            q_global, q_local, q_weights = self.get_features(img_tensors)

            # --- Stage 1: Global Search ---
            # Search in the LOADED (Train) Index
            D, I = self.index.search(q_global.cpu().numpy().astype('float32'), top_k_global)
            candidate_indices = I[0]

            # --- Stage 2: Local Rerank ---
            cand_locals = []
            valid_indices = []

            for idx in candidate_indices:
                if idx == -1: continue
                # No need to skip self, as test data is not in training index
                data = self.local_features[idx]
                cand_locals.append(data["local"].to(self.device).float())
                valid_indices.append(idx)
            
            if not cand_locals: continue

            # Reranking Logic
            t_tensor = torch.stack(cand_locals) # (K, L, 128)
            
            # Normalize
            q_norm = F.normalize(q_local, dim=-1) # (1, L, 128)
            t_norm = F.normalize(t_tensor, dim=-1) # (K, L, 128)
            
            # Similarity Matrix
            # q_norm (1, ...) -> expand to (K, ...)
            q_expand = q_norm.expand(len(cand_locals), -1, -1)
            sim_matrix = torch.bmm(q_expand, t_norm.transpose(1, 2))
            
            # Max Sim
            max_sim = sim_matrix.max(dim=-1).values
            
            # Weighted Sum
            w_expand = q_weights.expand(len(cand_locals), -1)
            scores = (max_sim * w_expand).sum(dim=-1)
            w_sum = w_expand.sum(dim=-1).clamp(min=1e-9)
            scores = scores / w_sum
            
            # Find Best
            scores = scores.detach().cpu().numpy()
            best_arg = np.argmax(scores)
            
            best_idx = valid_indices[best_arg]
            best_score = scores[best_arg]
            best_meta = self.metadata[best_idx]

            result_mapping[current_id] = {
                "report": item.get('report', ''),
                "similar_id": best_meta['id'],
                "similar_report": best_meta['report'],
                "score": float(best_score)
            }

        json_path = os.path.join(save_path, "test_similar_cases.json")
        with open(json_path, 'w') as f:
            json.dump(result_mapping, f, indent=2)
        print(f"Saved TEST similar case mapping to {json_path}")

    def search(self, query_img_path, top_k_global=200, top_k_rerank=10):
        """
        Performs the Two-Stage Retrieval using CMLIR logic.
        """
        if self.index is None:
            print("Index not loaded. Please build or load an index first.")
            return []

        # 1. Extract Query Features
        q_tensors = self.preprocess(query_img_path)
        if q_tensors is None: return []
        
        q_global, q_local, q_weights = self.get_features(q_tensors)
        
        # --- Stage 1: Global Retrieval (FAISS) ---
        D, I = self.index.search(q_global.cpu().numpy().astype('float32'), top_k_global)
        candidate_indices = I[0]
        
        # --- Stage 2: Local Rerank ---
        cand_locals = []
        valid_indices = []
        
        for idx in candidate_indices:
            if idx == -1: continue
            data = self.local_features[idx]
            cand_locals.append(data["local"].to(self.device).float())
            valid_indices.append(idx)
            
        if not cand_locals:
            return []
            
        t_tensor = torch.stack(cand_locals)
        q_tensor = q_local 
        q_w = q_weights    
        
        q_norm = F.normalize(q_tensor, dim=-1) 
        t_norm = F.normalize(t_tensor, dim=-1) 
        
        q_expand = q_norm.expand(len(cand_locals), -1, -1)
        sim_matrix = torch.bmm(q_expand, t_norm.transpose(1, 2))
        max_sim = sim_matrix.max(dim=-1).values 
        
        w_expand = q_w.expand(len(cand_locals), -1)
        scores = (max_sim * w_expand).sum(dim=-1)
        w_sum = w_expand.sum(dim=-1).clamp(min=1e-9)
        scores = scores / w_sum
        
        scores = scores.detach().cpu().numpy()
        sorted_idx = np.argsort(scores)[::-1][:top_k_rerank]
        
        results = []
        for i in sorted_idx:
            global_idx = valid_indices[i]
            meta_item = self.metadata[global_idx]
            score = scores[i]
            results.append({
                "path": meta_item['path'],
                "report": meta_item['report'],
                "score": float(score)
            })
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Stage Retrieval / Similar Case Generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to CMLIR checkpoint (.pth)")
    parser.add_argument("--annotation_file", type=str, default=None, help="Path to JSONL/JSON")
    parser.add_argument("--data_base_dir", type=str, default="", help="Base dir for images")
    parser.add_argument("--test_image", type=str, default=None, help="Query image path/dir (for test mode)")
    parser.add_argument("--save_path", type=str, default="retrieval_index", help="Index/Result save dir")
    parser.add_argument("--mode", type=str, default="build", choices=["build", "test", "simlar_case_creat", "create_test_similar"])
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Initialize Retriever
    retriever = TwoStageRetriever(args)
    
    # 1. BUILD INDEX / SIMILAR CASE CREATION (TRAIN on TRAIN)
    if args.mode in ["build", "simlar_case_creat"]:
        data_to_index = []
        valid_data_list = []
        if args.annotation_file and os.path.exists(args.annotation_file):
            print(f"Loading TRAINING data from: {args.annotation_file}")
            with open(args.annotation_file, 'r') as f:
                full_meta = json.load(f)
            
            if isinstance(full_meta, dict) and 'train' in full_meta:
                items = full_meta['train']
            elif isinstance(full_meta, list):
                items = full_meta
            else:
                items = []

            print(f"Loaded {len(items)} training items.")

            for item in items:
                report = item.get('report', '')
                img_paths = item.get('image_path', [])
                item_id = item.get('id', 'unknown')

                if not isinstance(img_paths, list):
                    img_paths = [img_paths]
                
                if not img_paths: continue
                
                full_img_paths = [os.path.join(args.data_base_dir, p) for p in img_paths]
                
                data_to_index.append({
                    'image_path': full_img_paths, 
                    'report': report, 
                    'id': item_id
                })
            
            # Load Valid
            valid_items = full_meta['val']
            print(f"Found {len(valid_items)} validation items in val split.")
            
            for item in valid_items:
                report = item.get('report', '')
                img_paths = item.get('image_path', [])
                item_id = item.get('id', 'unknown')
                full_img_paths = [os.path.join(args.data_base_dir, p) for p in img_paths]
                valid_data_list.append({'image_path': full_img_paths, 'report': report, 'id': item_id})

        if not data_to_index:
            print("No valid data found. Exiting.")
            sys.exit(1)
                
        # Build Index
        if args.mode == "build":
            retriever.build_index(data_to_index, args.save_path)
        else:
            retriever.load_index(args.save_path)
            retriever.create_similar_dataset(args.save_path, valid_data_list=valid_data_list)

    # 2. CREATE TEST SIMILAR (TEST on TRAIN)
    elif args.mode == "create_test_similar":
        # First, LOAD the existing TRAINING INDEX
        retriever.load_index(args.save_path)
        
        test_data_list = []
        if args.annotation_file and os.path.exists(args.annotation_file):
            print(f"Loading TEST data from: {args.annotation_file}")
            with open(args.annotation_file, 'r') as f:
                full_meta = json.load(f)
            
            # Extract TEST split
            if isinstance(full_meta, dict) and 'test' in full_meta:
                items = full_meta['test']
            else:
                print("Error: No 'test' split found in annotation file.")
                sys.exit(1)

            print(f"Loaded {len(items)} test items.")

            for item in items:
                # Same logic as before
                img_paths = item.get('image_path', [])
                item_id = item.get('id', 'unknown')
                
                if not isinstance(img_paths, list):
                    img_paths = [img_paths]
                if not img_paths: continue
                
                full_img_paths = [os.path.join(args.data_base_dir, p) for p in img_paths]
                
                test_data_list.append({
                    'image_path': full_img_paths, 
                    'id': item_id,
                    'report': item.get('report', '')
                })
        
        if not test_data_list:
            print("No valid test data found.")
            sys.exit(1)

        retriever.create_test_similar_dataset(test_data_list, args.save_path)

    # 2. TEST MODE
    elif args.mode == "test":
        if not args.test_image:
            print("Please provide --test_image for testing.")
            sys.exit(1)

        query_list = []
        if os.path.isdir(args.test_image):
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                query_list.extend(glob.glob(os.path.join(args.test_image, ext)))
            query_list.sort()
        elif os.path.exists(args.test_image):
            query_list.append(args.test_image)
            

        
        # Helper for MedCPT Sim (Optional Validation)
        print("Loading MedCPT for semantic validation...")
        try:
            medcpt_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
            medcpt_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(retriever.device)
            medcpt_model.eval()
        except Exception as e:
            print(f"Could not load MedCPT (Optional): {e}")
            medcpt_model = None

        def get_medcpt_emb(text_list):
            if medcpt_model is None: return None
            with torch.no_grad():
                inputs = medcpt_tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to(retriever.device)
                outputs = medcpt_model(**inputs)
                mask = inputs.attention_mask
                emb = (outputs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
                return F.normalize(emb, dim=-1)

        # Run Search        
        retriever.load_index(args.save_path)
        results = retriever.search(query_list, top_k_global=1000, top_k_rerank=10)
        
        # Dummy test report for similarity check (replace with real one if needed)
        # test_report = "the heart is normal in size . the mediastinum is unremarkable . the lungs are clear"
        # test_report = "A posterior-anterior view of a chest radiograph. The heart is enlarged with a cardiothoracic ratio of 54%. The lungs are hyperinflated. No focal lung lesion, consolidation, or pleural effusions are identified." 
        with open(os.path.join(args.test_image, "report.txt"), 'r') as rf:
            test_report = rf.read().strip()
            print(test_report)
        test_report_emb = get_medcpt_emb([test_report]) if medcpt_model else None
        
        print("\nTop Results (Reranked):")
        for rank, res in enumerate(results):
            print(f"{rank+1}. Path: {res['path']} (Score: {res['score']:.4f})")
            print(f"   Report: {res['report']}...") # Truncate long reports
            
            if test_report_emb is not None:
                retrieved_emb = get_medcpt_emb([res['report']])
                sim = (test_report_emb @ retrieved_emb.T).item()
                print(f"   [Validation] MedCPT Sim with dummy query: {sim:.4f}")