import os
import torch
import torchvision
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


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
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target < split_label]
        dataset.labels = dataset.labels[dataset.labels < split_label]
        dataset.data = dataset.data[np_target < split_label]
    else:
        dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
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
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target > split_label]
        dataset.labels = dataset.labels[dataset.labels > split_label]
        dataset.data = dataset.data[np_target > split_label]
    else:
        dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
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