import os
import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

class MazeDataset(Dataset):
    def __init__(self, root_dir, img_size=128, num_frames=16):
        self.root_dir = root_dir
        self.img_size = img_size
        self.num_frames = num_frames
        
        # 查找所有 mp4 文件
        self.video_files = sorted(glob.glob(os.path.join(root_dir, "maze3_*.mp4")))
        
        self.data_pairs = []
        for vid_path in self.video_files:
            basename = os.path.basename(vid_path)
            file_id = basename.split('.')[0] # maze3_0001
            img_filename = f"{file_id}_00.png"
            img_path = os.path.join(root_dir, img_filename)
            
            if os.path.exists(img_path):
                self.data_pairs.append((vid_path, img_path))
            else:
                print(f"Warning: Image not found for {vid_path}, skipping.")

    def __len__(self):
        return len(self.data_pairs)

    def _load_video_cv2(self, path):
        """使用 OpenCV 读取视频，替代 torchvision.read_video"""
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV 默认读取为 BGR，需要转为 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
            
        if len(frames) == 0:
            # 如果读取失败，返回全黑帧防止报错
            return torch.zeros(self.num_frames, self.img_size, self.img_size, 3, dtype=torch.uint8)

        # 转换为 numpy array: (T, H, W, C)
        video = np.stack(frames)
        
        # 简单的时间采样逻辑
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video = video[indices]
        else:
            # 如果视频太短，循环填充（Padding）
            pad_len = self.num_frames - total_frames
            padding = np.zeros((pad_len, *video.shape[1:]), dtype=video.dtype)
            video = np.concatenate([video, padding], axis=0)

        return torch.from_numpy(video)

    def __getitem__(self, idx):
        vid_path, img_path = self.data_pairs[idx]
        
        # 1. Load Video -> (T, H, W, C)
        video = self._load_video_cv2(vid_path)
        
        # 调整维度: (T, H, W, C) -> (C, T, H, W) 适配模型输入
        video = video.permute(3, 0, 1, 2) 
        
        # Resize Video
        # interpolate 要求输入为 (Batch, Channels, Time, Height, Width) 或 4D
        # 这里 video 是 (C, T, H, W)，Resize 会把它当做 (C, H, W) 处理（如果T看作Batch? 不对）
        # 正确做法：将 T 和 C 合并处理 spatial resize，或者使用 functional 逐帧处理
        # 简单起见，我们把 T 放在 Batch 维处理 resize
        
        C, T, H, W = video.shape
        # (C, T, H, W) -> (T, C, H, W)
        video = video.permute(1, 0, 2, 3) 
        
        # Resize: resize expect (..., H, W)
        video = F.resize(video, [self.img_size, self.img_size], interpolation=F.InterpolationMode.BILINEAR)
        
        # Back to (C, T, H, W)
        video = video.permute(1, 0, 2, 3)
        video = video.float() # [0, 255]
        
        # 2. Load Reference Image -> (C, H, W)
        # 使用 cv2 读取图片以保持一致性
        ref_img = cv2.imread(img_path)
        if ref_img is None:
             # Fallback black image
             ref_img = torch.zeros(3, self.img_size, self.img_size)
        else:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = torch.from_numpy(ref_img).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
            ref_img = F.resize(ref_img, [self.img_size, self.img_size], interpolation=F.InterpolationMode.BICUBIC)
            ref_img = ref_img.float()

        return video, ref_img