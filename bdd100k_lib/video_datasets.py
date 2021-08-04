from glob import glob
import os
import os.path as osp
import random
# from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data

from core.utils import frame_utils


class FlowBDDDataset(data.Dataset):
    def __init__(self):
        self.is_test = True
        self.init_seed = False
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return img1, img2, self.extra_info[index]

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class BDD(FlowBDDDataset):

    SUBSET_OPTIONS = ['train', 'val', 'test']
    DATASET_WEB = 'https://bdd-data.berkeley.edu/'

    def __init__(
        self,
        root,
        subset='train',
        debug_load_num=None,
        random_sample=True,
    ):
        super().__init__()
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')

        self.root = root
        self.subset = subset
        self.img_path = osp.join(root, 'images', subset)
        self._check_directories()

        # self.sequences = defaultdict(dict)

        dir_names = sorted(os.listdir(self.img_path))
        selected_dirs = dir_names
        all_datanum = len(dir_names)
        data_num = debug_load_num if debug_load_num else all_datanum
        data_num = data_num if data_num <= all_datanum else all_datanum
        if random_sample:
            selected_dirs = random.sample(dir_names, data_num)
        else:
            selected_dirs = dir_names[:data_num]
        for seq in selected_dirs:
            images = sorted(glob(osp.join(self.img_path, seq, '*.jpg')))
            # self.sequences[seq]['images'] = images
            for i in range(len(images) - 1):
                self.image_list += [[images[i], images[i + 1]]]
                self.extra_info += [(seq, i)]  # scene and frame_id

    def _check_directories(self):
        if not osp.exists(self.root) or not osp.exists(self.img_path):
            raise FileNotFoundError(f"BDD not found in the specified directory, \
                download it from {self.DATASET_WEB}")


#     def get_sequences(self):
#         for sequence in self.sequences:
#             images = [
#                 self.pil_to_tensor(Image.open(image_path))
#                 for image_path in self.sequences[sequence]['images']
#             ]
#             yield images, sequence
