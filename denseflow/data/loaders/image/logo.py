from denseflow.data.datasets.image import LogoData
from torchvision.transforms import Compose, ToTensor
from denseflow.data.transforms import Quantize
from denseflow.data import TrainTestLoader, DATA_PATH


class Logo(TrainTestLoader):
    """
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 64x64, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759
    """

    def __init__(
        self, root=DATA_PATH, num_bits=8, pil_transforms=[], test_transform=[]
    ):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = test_transform + [ToTensor(), Quantize(num_bits)]

        # Load data
        self.train = LogoData(root, train=True, transform=Compose(trans_train))
        self.test = LogoData(root, train=False, transform=Compose(trans_test))
