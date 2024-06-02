import os
import re
import os.path as osp
from glob import glob
import torch
from PIL import Image
from torch.utils.data import Dataset


class SYSUDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        root = osp.join(root, 'sysu')
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']
        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()
            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')
        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)
        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]
        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]
        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform
        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        path_HF = path.replace('sysu', 'sysu_HF')
        img = Image.open(path)
        img_HF = Image.open(path_HF)
        if self.transform is not None:
            img = self.transform(img)
            img_HF = self.transform(img_HF)
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)
        return img, img_HF, label, cam, path, item


class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))
        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]
        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)
        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths] 
        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.num_ids = num_ids
        self.transform = transform
        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)
        return img, label, cam, path, item

