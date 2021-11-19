from collections import defaultdict
from glob import glob
import os
import os.path as osp
import random
# from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

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
        data_start=0,
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
        selected_dirs = dir_names[data_start:]
        all_datanum = len(selected_dirs)
        data_num = debug_load_num if debug_load_num else all_datanum
        data_num = data_num if data_num <= all_datanum else all_datanum
        if random_sample:
            selected_dirs = random.sample(selected_dirs, data_num)
        else:
            selected_dirs = selected_dirs[:data_num]
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


class BDDVideo(data.Dataset):

    SUBSET_OPTIONS = ["train", "val", "test"]
    PHASE_OPTIONS = ["train", "evaluate"]
    DATASET_WEB = "https://bdd-data.berkeley.edu/"
    VOID_LABEL = 20
    PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60,
                                                               100], [0, 80, 100],
                        [0, 0, 230], [119, 11, 32], [0, 0, 0]]).astype(np.uint8)

    def __init__(self,
                 root,
                 subset="train",
                 phase="train",
                 data_start=-1,
                 num_frames=10,
                 normalize=False,
                 debug_mode=False,
                 debug_load_num=None):
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f"Subset should be in {self.SUBSET_OPTIONS}")
        if phase not in self.PHASE_OPTIONS:
            raise ValueError(f"Phase should be in {self.PHASE_OPTIONS}")

        self.root = root
        self.subset = subset
        self.phase = phase
        self.img_path = os.path.join(root, "images", subset)
        self.mask_path = os.path.join(root, "seg", "color_labels", subset)
        self.num_frames = num_frames
        self.debug_mode = debug_mode
        self.debug_load_num = debug_load_num
        self.config = f"bdd100k/{subset}"

        self.pil_to_tensor = lambda x: x
        if normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            trans_tmp = [transforms.Normalize(mean, std)]
            self.pil_to_tensor = transforms.Compose(trans_tmp)

        self.gt_frame = 60  # ground truth mask is given at 10th second

        self.sequences = defaultdict(dict)

        dir_names = sorted(os.listdir(self.img_path))
        selected_dirs = dir_names
        if debug_mode:
            datanum = debug_load_num if debug_load_num is not None else len(dir_names)
            e_idx = data_start + datanum
            selected_dirs = (random.sample(dir_names, datanum)
                             if data_start < 0 else dir_names[data_start:e_idx])
        for seq in selected_dirs:
            images = sorted(glob(os.path.join(self.img_path, seq, "*.jpg")))
            if len(images) < self.num_frames:
                continue
            self.sequences[seq]["images"] = images

        self.sequence_names = list(self.sequences.keys())

    def _check_directories(self):
        dw_msg = f"download it from {self.DATASET_WEB}"
        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"BDD not found in the specified directory, {dw_msg}")
        if self.phase == "evaluate" and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Annotations folder not found, {dw_msg}")

    def preprocessing_image_for_raft(self, image):
        image = np.array(image).astype(np.uint8)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def get_imgs(self, idx, s_frame, num_frames=None):
        if num_frames is None:
            num_frames = self.num_frames
        sequence = self.__getitem__(idx)
        images = []
        all_num_frames = len(sequence)
        for i in range(s_frame, min(s_frame + num_frames, all_num_frames)):
            image = Image.open(sequence[i])
            if image.height > image.width:
                image = image.rotate(90, expand=True)
            image = self.preprocessing_image_for_raft(image)
            images.append(image)

        inputs = [self.pil_to_tensor(image) for image in images]
        if self.debug_mode:
            # name = self.sequence_names[idx]
            # print(idx, ": ", name)
            inputs = [inputs, torch.tensor([idx, s_frame, num_frames])]
        return inputs

    def __getitem__(self, idx):
        sequence = self.sequences[self.sequence_names[idx]]
        return sequence["images"]

    def __len__(self):
        return len(self.sequences)

    def _get_info(self, info):
        index = info[0]
        name = self.sequence_names[index]
        str_info = "index {} name {} s_frame {} num_frames {}".format(
            index, name, info[1], info[2])
        return str_info

    def get_info(self, info):
        dim = info.dim()
        if dim < 2:
            info = info.unsqueeze(0)
        tmp = info.tolist()
        info_list = ""
        for t in tmp:
            info_list += self._get_info(t) + "\n"
        return info_list

    def get_sequences(self):
        if self.phase == "train":
            for sequence in self.sequences:
                images = [
                    self.pil_to_tensor(Image.open(image_path))
                    for image_path in self.sequences[sequence]["images"]
                ]
                yield images, sequence
        else:
            for sequence in self.sequences_with_mask:
                images = [
                    self.pil_to_tensor(Image.open(image_path))
                    for image_path in self.sequences[sequence]["images"][self.gt_frame:]
                ]
                mask = [
                    np.asarray(Image.open(mask_path)) if mask_path is not None else None
                    for mask_path in self.sequences[sequence]["masks"][self.gt_frame:]
                ]
                yield images, mask, sequence
