import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import cv2 

def _fuse_images(images_pil, method='direct'):
    """
    Fuse three single-channel PIL images into a three-channel RGB image
    Args:
        images_pil (list): A list containing three single-channel PIL.Image objects.
        method (str): Fusion method. Optional:
                      'RGB_0_1_2': img1->R, img2->G, img3->B
                      'RGB_1_0_2': img1->G, img2->R, img3->B
                      'RGB_1_2_0': img1->B, img2->R, img3->G
                      'hls_0_1_2': img1->H, img2->L, img3->S 
                      'hls_1_0_2': img1->L, img2->H, img3->S
                      'hls_1_2_0': img1->S, img2->H, img3->L 
                      'ycbcr_0_1_2': img1->Y, img2->Cb, img3->Cr 
    Returns:
        PIL.Image: 融合后的三通道 RGB PIL 图像。
    """
    if len(images_pil) != 3:
        raise ValueError(f"需要三张图像进行融合，但收到了 {len(images_pil)} 张。")
        
    # img1=S0， img2=L1， img3=L2
    img1, img2, img3 = [np.array(img) for img in images_pil]

    if method == 'rgb_0_1_2':
        rgb_array = np.stack([img1, img2, img3], axis=-1)
        return Image.fromarray(rgb_array, 'RGB')
    elif method == 'rgb_1_0_2':
        rgb_array = np.stack([img2, img1, img3], axis=-1)
        return Image.fromarray(rgb_array, 'RGB')
    elif method == 'rgb_1_2_0':
        rgb_array = np.stack([img2, img3, img1], axis=-1)
        return Image.fromarray(rgb_array, 'RGB')
        
    elif method == 'hls_0_1_2': 
        H = (img1.astype(np.float32) / 255.0 * 179).astype(np.uint8)
        L = img2.astype(np.uint8)
        S = img3.astype(np.uint8)
        hls_array = np.stack([H, L, S], axis=-1)
        bgr_array = cv2.cvtColor(hls_array, cv2.COLOR_HLS2BGR)
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array, 'RGB')

    elif method == 'hls_1_0_2':
        H = (img2.astype(np.float32) / 255.0 * 179).astype(np.uint8)
        L = img1.astype(np.uint8)
        S = img3.astype(np.uint8)
        hls_array = np.stack([H, L, S], axis=-1)
        bgr_array = cv2.cvtColor(hls_array, cv2.COLOR_HLS2BGR)
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array, 'RGB')
    
    elif method == 'hls_1_2_0':
        H = (img2.astype(np.float32) / 255.0 * 179).astype(np.uint8)
        L = img3.astype(np.uint8)
        S = img1.astype(np.uint8)
        hls_array = np.stack([H, L, S], axis=-1)
        bgr_array = cv2.cvtColor(hls_array, cv2.COLOR_HLS2BGR)
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array, 'RGB')
        
    elif method == 'ycbcr_0_1_2': 
        Y = img1.astype(np.uint8)
        Cb = img2.astype(np.uint8)
        Cr = img3.astype(np.uint8)  
        ycrbr_array = np.stack([Y, Cr, Cb], axis=-1)
        rgb_array = cv2.cvtColor(ycrbr_array, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(rgb_array, 'RGB')

    else:
        raise NotImplementedError(f"fusion method '{method}' is not implemented." )


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, fusion_method='direct'):
        """
        Args:
            root_dir (str): 包含类别子目录的数据集根目录。
            transform (callable, optional): 应用于样本的变换。
            fusion_method (str): Methods for fusing three grayscale images
                                 option: 'direct', 'hls_0_1_2', 'hls_1_0_2', 'hls_1_2_0','ycbcr_0_1_2'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.fusion_method = fusion_method 

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            try:
                file_list = os.listdir(cls_dir)
                if not file_list: continue
                # The file name format {base_name}_0.jpg, {base_name}_1.jpg, {base_name}_2.jpg
                base_files = set(f.rsplit('_', 1)[0] for f in file_list if '_' in f and f.lower().endswith(('.jpg', '.png', '.bmp')))
            except OSError as e:
                print(f"Warning: Unable to access or read the directory {cls_dir}: {e}")
                continue
        
            for base in base_files:
                paths = [
                    os.path.join(cls_dir, f"{base}_0.jpg"),
                    os.path.join(cls_dir, f"{base}_1.jpg"),
                    os.path.join(cls_dir, f"{base}_2.jpg")
                ]
                if all(os.path.exists(p) for p in paths):
                    item = (paths, self.class_to_idx[cls], base)
                    self.samples.append(item)

        if not self.samples:
            raise ValueError(f"Error:directory '{root_dir}' no valid trio samples")

        self.labels = np.array([label for _, label, _ in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, base_name = self.samples[idx]
        
        # 1. 加载三张灰度图为 PIL Image 对象
        try:
            imgs_pil = [Image.open(p).convert('L') for p in paths]
        except Exception as e:
            print(f"错误: 加载索引 {idx} 的图像失败，路径: {paths}")
            raise e

        # 2. 将三张灰度图融合成一张三通道 RGB PIL 图像
        fused_pil_image = _fuse_images(imgs_pil, method=self.fusion_method)
        if self.transform:
            fused_pil_image = self.transform(fused_pil_image)
        return fused_pil_image, label, base_name

    def get_labels(self):
        return self.labels
