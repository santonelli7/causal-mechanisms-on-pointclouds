import random
import os, urllib, shutil
from mlxtend.data import loadlocal_mnist
from tqdm.autonotebook import tqdm
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

from src.datasets.data import TransformedSample
from src.utils import img_to_point_cloud, extract_gz

class MNISTPointCloud(InMemoryDataset):

    urls = {"train": ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"],
            "t10k": ["http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
            }

    def __init__(self, root, transforms, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.idx_to_transf = {idx: transf for idx, transf in enumerate(transforms)} 
        self.transf_to_idx = {transf: idx for idx, transf in self.idx_to_transf.items()}
        super(MNISTPointCloud, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36')
        for key, url in self.urls.items():
            imgs_filename = f'{self.raw_dir}/{key}-images-idx3-ubyte'
            labels_filename = f'{self.raw_dir}/{key}-labels-idx1-ubyte'
            opener.retrieve( url[0], f'{self.root}/{key}-images-idx3-ubyte.gz')
            opener.retrieve( url[1], f'{self.root}/{key}-labels-idx1-ubyte.gz')
            extract_gz(f'{self.root}/{key}-images-idx3-ubyte.gz', imgs_filename)
            extract_gz(f'{self.root}/{key}-labels-idx1-ubyte.gz', labels_filename)
        
            os.remove(f'{self.root}/{key}-images-idx3-ubyte.gz')
            os.remove(f'{self.root}/{key}-labels-idx1-ubyte.gz')
        
        metadata_folder = os.path.join(self.root, '__MACOSX')
        if os.path.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        dataset = 't10k' if dataset == 'test' else dataset
        # Read data into huge `Data` list.
        X, y = loadlocal_mnist(images_path=f'{self.raw_dir}/{dataset}-images-idx3-ubyte', labels_path=f'{self.raw_dir}/{dataset}-labels-idx1-ubyte')
        normalize = T.NormalizeScale()

        data_list = []
        shape = (28, 28)
        Q_n = {idx: [] for idx, _ in self.idx_to_transf.items()}
        for image, label in tqdm(zip(X, y), total=len(X)):
            pos, face = img_to_point_cloud(image.reshape(shape[0], shape[1]))
            y = torch.tensor([label])
            data = normalize(Data(pos=pos, face=face, y=y))
            data = TransformedSample(pos=data.pos, face=data.face, y=data.y, original_pos=data.pos)
            for idx, transf in self.idx_to_transf.items():
                transf_data = transf(data.clone())
                if self.pre_transform is not None:
                    transf_data = self.pre_transform(transf_data)
                Q_n[idx].append(transf_data)
            data_list.append(data)
        
        new_data_list = []
        for data in data_list:
            t = random.sample(list(self.idx_to_transf.keys()), 1)[0]
            transf_sample = random.sample(Q_n[t], 1)[0]
            data = TransformedSample(pos=data.pos, face=data.face, y=data.y, original_pos=data.original_pos, pos_transf=transf_sample.pos, face_transf=transf_sample.face, y_transf=transf_sample.y, original_pos_transf=transf_sample.original_pos, transf=t)
            new_data_list.append(data)
        data_list = new_data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return self.collate(data_list)
    