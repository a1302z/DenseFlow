import math
from torch.utils.data import DataLoader
from torchvision.transforms import (
    RandomHorizontalFlip,
    Pad,
    RandomAffine,
    CenterCrop,
    RandomCrop,
    Resize,
)
from denseflow.data.loaders.image import (
    CIFAR10,
    ImageNet32,
    ImageNet64,
    SVHN,
    MNIST,
    CIFAR10Supervised,
    SVHNSupervised,
    CIFAR100Supervised,
    CelebA,
    Logo,
)

dataset_choices = {
    "cifar10",
    "cifar10sup",
    "cifar100sup",
    "imagenet32",
    "imagenet64",
    "svhn",
    "svhnsup",
    "mnist",
    "celeba",
    "logo",
}


def add_data_args(parser):

    # Data params
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=dataset_choices
    )
    parser.add_argument("--num_bits", type=int, default=8)

    # Train params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=eval, default=True)
    parser.add_argument("--augmentation", type=str, default=None)


def get_data_id(args):
    return "{}_{}bit".format(args.dataset, args.num_bits)


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    data_shape = get_data_shape(args.dataset)
    pil_transforms, test_transform = get_augmentation(
        args.augmentation, args.dataset, data_shape
    )
    if args.dataset == "cifar10":
        dataset = CIFAR10(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "celeba":
        dataset = CelebA(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "imagenet32":
        dataset = ImageNet32(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "imagenet64":
        dataset = ImageNet64(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "svhn":
        dataset = SVHN(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "svhnsup":
        dataset = SVHNSupervised(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "mnist":
        dataset = MNIST(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == "logo":
        dataset = Logo(
            num_bits=args.num_bits,
            pil_transforms=pil_transforms,
            test_transform=test_transform,
        )

    # Data Loader
    train_loader = DataLoader(
        dataset.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    eval_loader = DataLoader(
        dataset.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return train_loader, eval_loader, data_shape


def get_augmentation(augmentation, dataset, data_shape):
    c, h, w = data_shape

    pil_transforms = [
        Resize(h),
        RandomCrop(h),
    ]
    test_transform = [
        Resize(h),
        CenterCrop(h),
    ]
    if augmentation is None:
        pass
    elif augmentation == "horizontal_flip":
        pil_transforms += [RandomHorizontalFlip(p=0.5)]
    elif augmentation == "neta":
        assert h == w
        pil_transforms = [
            Pad(int(math.ceil(h * 0.04)), padding_mode="edge"),
            RandomAffine(degrees=0, translate=(0.04, 0.04)),
            Resize(h),
            RandomCrop(h),
        ]
    elif augmentation == "eta":
        assert h == w
        pil_transforms = [
            RandomHorizontalFlip(),
            Pad(int(math.ceil(h * 0.04)), padding_mode="edge"),
            RandomAffine(degrees=0, translate=(0.04, 0.04)),
            Resize(h),
            RandomCrop(h),
        ]
    # return pt
    return pil_transforms, test_transform


def get_data_shape(dataset):
    if dataset == "cifar10" or dataset == "cifar10sup" or dataset == "cifar100sup":
        return (3, 32, 32)
    elif dataset == "imagenet32":
        return (3, 32, 32)
    elif dataset == "imagenet64" or dataset == "celeba" or dataset == "logo":
        return (3, 64, 64)
    elif dataset == "svhn" or dataset == "svhnsup":
        return (3, 32, 32)
    elif dataset == "mnist":
        return (1, 28, 28)
