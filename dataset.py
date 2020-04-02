
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF


class Vaihgen(Dataset):
    def __init__(self):
        self.samples = [1, 2, 3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.samples[idx], self.samples[idx]]


class Data(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.file_triples = []
        self.cache_triples = []
        self.data_dir = data_dir
        self.mode = mode
        self.read_triples()

    def read_triples(self):
        triple_file = None
        if self.mode == 'train':
            triple_file = 'train.txt'
        elif self.mode == 'val':
            triple_file = 'val.txt'
        if triple_file is None:
            raise ValueError("The mode only surport train or val")

        triple_file = os.path.join(self.data_dir, triple_file)
        with open(triple_file, 'r') as file:
            line = file.readline()
            while line:
                line = line.strip()
                if not line.startswith('#'):
                    triple = line.split()
                    self.file_triples.append((triple[0], triple[1], triple[2]))
                    self.cache_triples.append(None)
                line = file.readline()

    def __len__(self):
        return len(self.file_triples)

    def __getitem__(self, idx):

        triple = self.file_triples[idx]
        color_file = os.path.join(self.data_dir, 'color', triple[0])
        label_file = os.path.join(self.data_dir, 'label', triple[1])
        depth_file = os.path.join(self.data_dir, 'height', triple[2])

        cache_triple = self.cache_triples[idx]
        if cache_triple is None:
            color_image = Image.open(color_file)
            label_image = Image.open(label_file)
            depth_image = Image.open(depth_file)
            depth_image = TF.to_grayscale(depth_image)

            color_image = TF.resize(color_image, (512, 512))
            label_image = TF.resize(label_image, (512, 512))
            depth_image = TF.resize(depth_image, (512, 512))

            self.cache_triples[idx] = (color_image, label_image, depth_image)
        else:
            color_image = cache_triple[0]
            label_image = cache_triple[1]
            depth_image = cache_triple[2]

        # crop_paras = RandomCrop.get_params(color_image, output_size=(512, 512))
        # color_image = TF.crop(color_image, *crop_paras)
        # label_image = TF.crop(label_image, *crop_paras)
        # depth_image = TF.crop(depth_image, *crop_paras)

        color_image = TF.to_tensor(color_image)
        label_image = TF.to_tensor(label_image)
        depth_image = TF.to_tensor(depth_image)

        return (color_image, label_image, depth_image)

