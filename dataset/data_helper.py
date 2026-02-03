
import os
import json
from pyexpat import features
import re
import pandas as pd
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

        # --- 新增：加载 ROCO 挖掘好的负样本索引 ---
        if self.dataset == 'roco':
            # 假设你把这两个字典存成了 json
            import json
            roco_hn_path = os.path.join(args.base_dir, "hard_negatives.json")
            train_captions_path = os.path.join(args.base_dir, "train_captions.csv")
            df = pd.read_csv(train_captions_path)
            self.id2caption = dict(zip(df['ID'], df['Caption'].fillna("")))
            with open(roco_hn_path, 'r') as f:
                self.roco_hn_dict = json.load(f)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        elif self.dataset == "mimic_cxr":
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        else:
            report = report.strip().lower()
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        curr_id = str(features.get('id', ''))
        to_return = {'id': curr_id}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        # --- ROCO 特殊处理逻辑 ---
        if self.dataset == 'roco' and getattr(self.args, 'retrieval_only', False):
            # 1. 准备文本列表 [Pos, Neg1, Neg2, ...]
            neg_ids = self.roco_hn_dict.get(curr_id, [])[:15] # 取前15个
            all_texts = [report] + [self.id2caption.get(nid, "") for nid in neg_ids]
            to_return['input_text'] = all_texts # 此时是 List[str]
            
            # 2. 准备图片列表 [Pos_Img, Neg1_Img, Neg2_Img, ...]
            images = []
            # 加载正样本
            pos_path = os.path.join(self.args.base_dir, features['image_path'][0])
            with Image.open(pos_path) as pil:
                pil = pil.convert('RGB')
                images.append(self._parse_image(pil))
            
            # 加载负样本图片
            for nid in neg_ids:
                # 注意路径拼接逻辑需与你硬盘存储一致
                path = os.path.join(self.args.base_dir, "train", f"{nid}.jpg")
                with Image.open(path) as pil:
                    pil = pil.convert('RGB')
                    images.append(self._parse_image(pil))

            
            to_return["image"] = torch.stack(images) # (16, 3, H, W)
            return to_return

        # --- 原始数据集逻辑 (Iuxray/Mimic) ---
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset
