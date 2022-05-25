# 对每个序列，返回所有图片和标注文件
import os
import json
import cv2 as cv
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as transforms

class YTB_Test(data.Dataset):
    def __init__(self, root, anno_path='Annotations', meta_path='meta18.json'):
        self.img_dir = os.path.join(root, 'valid', 'JPEGImages')
        self.anno_dir = os.path.join(root, 'valid', anno_path)
        meta_path = os.path.join(root, 'valid', meta_path)
        with open(meta_path) as f:
            self.meta_data = json.load(f)['videos']
        self.seq_list = list(self.meta_data.keys())

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq_name = self.seq_list[idx]
        seq_img_dir = os.path.join(self.img_dir, seq_name)
        seq_anno_dir = os.path.join(self.anno_dir, seq_name)
        seq_imgs = os.listdir(seq_img_dir)
        seq_annos = os.listdir(seq_anno_dir)
        seq_imgs.sort()
        seq_annos.sort()
        for i, img_name in enumerate(seq_imgs):
            if img_name.replace('.jpg', '.png') == seq_annos[0]:
                break
        seq_imgs = seq_imgs[i:]

        objs = []
        imgs, annos, new_objs = [], [], []
        for img_name in seq_imgs:
            img_path = os.path.join(seq_img_dir, img_name)
            img = cv.imread(img_path)
            img = np.float32(img) / 255.0
            img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([50,0,0], [50,127,127])(img)
            imgs.append(img)
            if img_name.replace('.jpg', '.png') in seq_annos:
                anno_path = os.path.join(seq_anno_dir, img_name.replace('.jpg', '.png'))
                anno = np.expand_dims(np.atleast_3d(Image.open(anno_path))[..., 0], 0)
                obj_ids = np.sort(np.unique(anno))[1:]
                new_obj = []
                for obj_id in obj_ids:
                    if obj_id not in objs:
                        objs.append(obj_id)
                        new_obj.append(obj_id)
                anno = torch.Tensor(anno).contiguous().long()
                annos.append(anno)
                new_objs.append(new_obj)
            else:
                annos.append(torch.zeros(1))
                new_objs.append([])

        return imgs, annos, new_objs, {"seq_name":seq_name, "seq_imgs":seq_imgs}
            
