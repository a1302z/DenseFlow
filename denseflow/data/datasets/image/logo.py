import os
import torch.utils.data as data
from PIL import Image
from random import sample, seed
from denseflow.data import DATA_PATH


seed(0)


class LogoData(data.Dataset):
    def __init__(
        self,
        root=DATA_PATH,
        folder="LLD-logo-files",
        train=True,
        transform=None,
        train_split=0.8,
    ):
        self.root = os.path.join(os.path.expanduser(root), folder)
        self.train = train
        self.transform = transform

        self.files = [os.path.join(self.root, file) for file in os.listdir(self.root)]
        N_files = len(self.files)
        N_samples = (
            round(train_split * N_files)
            if train
            else N_files - round(train_split * N_files)
        )
        idcs = sample(range(N_files), N_samples)
        self.files = [self.files[i] for i in idcs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """

        img = Image.open(self.files[index])

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)

