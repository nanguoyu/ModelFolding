import os
import torch
import torchvision
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import collections

def plot_class_distribution(dataset, dataset_name):
    class_counts = collections.Counter(dataset.labels)
    class_labels = [str(i) for i in range(10)]

    df = pd.DataFrame(list(class_counts.items()), columns=['Class Index', 'Sample Count'])
    df['Class Name'] = df['Class Index'].map(lambda x: class_labels[x])

    df = df.sort_values(by='Sample Count', ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df['Class Name'], df['Sample Count'])
    plt.xticks(rotation=0)
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.title(f'{dataset_name} Class Distribution')
    plt.savefig(f'{dataset_name}_class_distribution.png')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class BalancedDataset(Dataset):
    def __init__(self, dataset, target_count=None):
        """
        :param dataset: Original dataset
        :param target_count: Target number of samples per class (default is the mean number of samples across all classes)
        """
        super().__init__()

        # Ensure compatibility with different versions of torchvision
        if hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
            self.label_attr = 'labels'
        elif hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
            self.label_attr = 'targets'
        else:
            raise AttributeError("Dataset does not have 'labels' or 'targets' attribute.")

        self.dataset = dataset
        self.original_indices = np.arange(len(dataset))

        # Compute class sample counts
        class_counts = np.bincount(labels)
        num_classes = len(class_counts)

        # Compute target number of samples per class (default: mean sample count)
        if target_count is None:
            target_count = int(np.mean(class_counts))

        # Generate balanced indices
        balanced_indices = []
        for cls in range(num_classes):
            cls_indices = np.where(labels == cls)[0]

            # Oversample or undersample
            if len(cls_indices) < target_count:
                sampled_indices = np.random.choice(cls_indices, target_count, replace=True)
            else:
                sampled_indices = np.random.choice(cls_indices, target_count, replace=False)

            balanced_indices.extend(sampled_indices)

        # Shuffle indices
        np.random.shuffle(balanced_indices)

        # Store balanced dataset indices
        self.indices = balanced_indices
        self.labels = labels[self.indices]  # Balanced labels
        if hasattr(dataset, 'data'):
            self.data = dataset.data[self.indices]  # Filter dataset data if it exists

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]  # Get original dataset index
        return self.dataset[original_idx]  # Return sample from the original dataset


def get_cifar100(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR100(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        mnistTrainSet,
        batch_size=bs,
        shuffle=True,
        num_workers=8)
    
    return loader


def get_imagenet(datadir, train=True, bs=256):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    get_dataset = getattr(datasets, "ImageNet")

    normalize = torchvision.transforms.Normalize(mean=mean, std=std)
    tr_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), normalize])
    val_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), normalize])
    if train is True:
        dataset = get_dataset(root=datadir, split='train', transform=tr_transform)
    else:
        dataset = get_dataset(root=datadir, split='val', transform=val_transform)

    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)

    return data_loader


def get_cifar10(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(15), torchvision.transforms.ToTensor(), normalize])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        
    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR10(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        mnistTrainSet,
        batch_size=bs, #256
        shuffle=True,
        num_workers=8)
    
    return loader


def get_cifar10_split_a(train=True, bs=512):
    get_dataset = getattr(datasets, "CIFAR10")
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2470, 0.2435, 0.2616]
    split_label = 5
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target < split_label]
        dataset.targets = dataset.targets[dataset.targets < split_label]
        dataset.data = dataset.data[np_target < split_label]
    else:
        dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target < split_label]
        dataset.targets = dataset.targets[dataset.targets < split_label]
        dataset.data = dataset.data[np_target < split_label]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)
    print("Using PyTorch dataset.")
    return data_loader

def get_cifar10_split_b(train=True, bs=512):
    get_dataset = getattr(datasets, "CIFAR10")
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2470, 0.2435, 0.2616]
    split_label = 4
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target > split_label]
        dataset.targets = dataset.targets[dataset.targets > split_label]
        dataset.data = dataset.data[np_target > split_label]
    else:
        dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target > split_label]
        dataset.targets = dataset.targets[dataset.targets > split_label]
        dataset.data = dataset.data[np_target > split_label]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)
    print("Using PyTorch dataset.")
    return data_loader

def get_svhn(train=True, bs=512):
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2470, 0.2435, 0.2616]
    get_dataset = getattr(datasets, "SVHN")
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'
    
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
    else:
        dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)
    return data_loader

def get_svhn_split_a(train=True, bs=512):
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2470, 0.2435, 0.2616]
    get_dataset = getattr(datasets, "SVHN")
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'

    split_label = 5
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        dataset = BalancedDataset(dataset)
        plot_class_distribution(dataset, "SVHN Split A Train")
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target < split_label]
        dataset.labels = dataset.labels[dataset.labels < split_label]
        dataset.data = dataset.data[np_target < split_label]
    else:
        dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
        dataset = BalancedDataset(dataset)
        plot_class_distribution(dataset, "SVHN Split A Test")
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target < split_label]
        dataset.labels = dataset.labels[dataset.labels < split_label]
        dataset.data = dataset.data[np_target < split_label]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)

    return data_loader

def get_svhn_split_b(train=True, bs=512):
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2470, 0.2435, 0.2616]
    get_dataset = getattr(datasets, "SVHN")
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'

    split_label = 4
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        dataset = BalancedDataset(dataset)
        plot_class_distribution(dataset, "SVHN Split B Train")
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target > split_label]
        dataset.labels = dataset.labels[dataset.labels > split_label]
        dataset.data = dataset.data[np_target > split_label]
    else:
        dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
        dataset = BalancedDataset(dataset)
        plot_class_distribution(dataset, "SVHN Split B Test")
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target > split_label]
        dataset.labels = dataset.labels[dataset.labels > split_label]
        dataset.data = dataset.data[np_target > split_label]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)
    return data_loader

def get_cifar100(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(15), torchvision.transforms.ToTensor(), normalize])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        
    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR100(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        mnistTrainSet,
        batch_size=bs, #256
        shuffle=True,
        num_workers=8)
    
    return loader

def get_cifar100_split_a(train=True, bs=512):
    get_dataset = getattr(datasets, "CIFAR100")
    mean=[0.5071, 0.4865, 0.4409]
    std=[0.2673, 0.2564, 0.2762]
    split_label = 50
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target < split_label]
        dataset.targets = dataset.targets[dataset.targets < split_label]
        dataset.data = dataset.data[np_target < split_label]
    else:
        dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target < split_label]
        dataset.targets = dataset.targets[dataset.targets < split_label]
        dataset.data = dataset.data[np_target < split_label]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)
    print("Using PyTorch dataset.")
    return data_loader

def get_cifar100_split_b(train=True, bs=512):
    get_dataset = getattr(datasets, "CIFAR100")
    mean=[0.5071, 0.4865, 0.4409]
    std=[0.2673, 0.2564, 0.2762]
    split_label = 49
    datadir = os.path.dirname(os.path.abspath(__file__)) + '/data'
    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if train:
        dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target > split_label]
        dataset.targets = dataset.targets[dataset.targets > split_label]
        dataset.data = dataset.data[np_target > split_label]
    else:
        dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target > split_label]
        dataset.targets = dataset.targets[dataset.targets > split_label]
        dataset.data = dataset.data[np_target > split_label]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)
    print("Using PyTorch dataset.")
    return data_loader