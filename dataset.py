import random
import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T

class CumstomDataset(data.Dataset):
    def __init__(self, phase = 'train', data_dir = './data/'):
        self.v_trans = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        
        self.t_trans = T.Compose([
                T.CenterCrop(320),
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

        # self.t_augment = T.Compose([
        #         T.RandomResizedCrop((256, 256)),
        #         T.RandomHorizontalFlip(0.2),
        #         T.RandomVerticalFlip(0.2),
        #         T.ToTensor(),
        #     ])


        self.touch_paths = [f for f in glob.glob(data_dir + 'touch/' + phase + '/*')]
        self.vision_paths = [f for f in glob.glob(data_dir + 'vision/' + phase + '/*')]
        self.obj_ids = [l.split('/')[-1].split('-')[0] for l in self.touch_paths]

        assert len(self.touch_paths) == len(self.vision_paths)

    def __len__(self):
        return len(self.touch_paths)

    def __getitem__(self, idx):
        t_path = self.touch_paths[idx]
        t_image = Image.open(t_path).convert('RGB')
        t_image = self.t_trans(t_image)

        v_path = self.vision_paths[idx]
        v_image = Image.open(v_path).convert('RGB')
        v_image = self.v_trans(v_image)

        
        label = 1
        if random.random() < 0.3:
            _idx = random.randint(idx,len(self.touch_paths)-4) \
                if idx<len(self.touch_paths)-4 else random.randint(0, idx-4)
            t_path = self.touch_paths[_idx]
            t_image = Image.open(t_path).convert('RGB')
            t_image = self.t_trans(t_image)
            label = 0
        
        return t_image, v_image, label, self.obj_ids[idx]