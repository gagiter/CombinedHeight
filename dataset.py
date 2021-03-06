
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
    def __init__(self, data_dir, size=512, mode='train', device='cpu'):
        self.file_triples = []
        self.cache_triples = []
        self.data_dir = data_dir
        self.mode = mode
        self.size = size
        self.device = device
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
        depth_file = os.path.join(self.data_dir, 'height', triple[2])

        cache_triple = self.cache_triples[idx]
        if cache_triple is None:
            color_image = Image.open(color_file)
            depth_image = Image.open(depth_file)
            depth_image = TF.to_grayscale(depth_image)

            color_image = TF.resize(color_image, self.size * 2)
            depth_image = TF.resize(depth_image, self.size * 2)
            self.cache_triples[idx] = (color_image, depth_image)
        else:
            color_image = cache_triple[0]
            depth_image = cache_triple[1]

        crop_paras = RandomCrop.get_params(color_image, output_size=(self.size, self.size))
        color_image = TF.crop(color_image, *crop_paras)
        depth_image = TF.crop(depth_image, *crop_paras)

        color_image = TF.to_tensor(color_image).to(self.device)
        depth_image = TF.to_tensor(depth_image).to(self.device)

        item = {'color': color_image, 'depth': depth_image}
        return item

