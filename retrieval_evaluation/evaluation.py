import json
import numpy as np

def evaluate_kis(benchmark_path, prediction_path):
    # 1. 加载数据
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(prediction_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # 将预测结果转为字典方便查询
    pred_dict = {item['id']: item['scores'] for item in pred_data}
    
    ranks = []
    
    print(f"开始评测... 总样本数: {len(gt_data)}")
    
    for item in gt_data:
        query_id = item['id']
        gold_id = item['gold']
        
        if query_id not in pred_dict:
            print(f"警告: Query {query_id} 缺失预测结果，跳过。")
            continue
            
        scores = pred_dict[query_id]
        
        # 提取当前 Query 下的 10 个图像及其得分 (1个Gold + 9个Candidates)
        # 注意：评测只针对这 10 个候选项进行重排
        target_ids = [gold_id] + item['candidates']
        
        # 过滤得分，只保留这 10 个图像的得分
        filtered_scores = {tid: scores.get(tid, -1e9) for tid in target_ids}
        
        # 按照得分从高到低排序
        sorted_ids = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_list = [x[0] for x in sorted_ids]
        
        # 找到 Gold 样本的排名 (1-indexed)
        rank = sorted_list.index(gold_id) + 1
        ranks.append(rank)

    # 2. 计算指标
    ranks = np.array(ranks)
    num_queries = len(ranks)
    
    r1 = np.sum(ranks <= 1) / num_queries
    r5 = np.sum(ranks <= 5) / num_queries
    mrr = np.mean(1.0 / ranks)
    medr = np.median(ranks)
    
    # 3. 打印结果
    print("-" * 30)
    print(f"📊 评测结果 (KIS Benchmark)")
    print("-" * 30)
    print(f"Recall @ 1: {r1*100:>6.2f}%")
    print(f"Recall @ 5: {r5*100:>6.2f}%")
    print(f"MRR:        {mrr:>8.4f}")
    print(f"Median Rank:{medr:>8.1f}")
    print("-" * 30)

    return {"R@1": r1, "R@5": r5, "MRR": mrr, "MedR": medr}

if __name__ == "__main__":
    evaluate_kis('rocov2_48x10_kis_benchmark.json', 'prediction/scmlir_img2img_result.json')