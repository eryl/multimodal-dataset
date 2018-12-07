import torch
import torch.utils.data

import multimodal.dataset.video

class TorchVideoDataset(torch.utils.data.Dataset):
    def __init__(self, *video_stores):
        self.video_stores = video_stores
        