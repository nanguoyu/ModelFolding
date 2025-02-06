from model.resnet import ResNet50, fuse_channel_resnet50_clustering
from utils.utils import load_model, eval_model
from utils.datasets import get_cifar100, get_cifar100_split_a, get_cifar100_split_b

import argparse
import torch
import copy
from tqdm import tqdm
from thop import profile


def test_merge(origin_model_a, checkpoint_a, origin_model_b, checkpoint_b, test_loader, train_loader, method):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model_a.cuda()
    origin_model_a.eval()
    origin_model_b.cuda()
    origin_model_b.eval()

    _, origin_param = profile(origin_model_a, inputs=(input,))
    model, _ = method(
        origin_model_a, 
        origin_model_b, 
        checkpoint_a, 
        checkpoint_b, 
        regularizer=1.0
    )

    model.cuda()
    model.eval()
    _, param = profile(model, inputs=(input,))
    
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None

    model.train()
    for x, _ in tqdm(train_loader, desc="REPAIR"):
        model(x.to("cuda"))

    model.eval()
    acc, _ = eval_model(model, test_loader)

    print(f"model after adapt: acc:{acc * 100:.2f}")

    return acc, param / origin_param, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", type=str)
    parser.add_argument("--checkpoint_b", type=str)

    args = parser.parse_args()

    model_a = ResNet50(num_classes=100)
    model_b = ResNet50(num_classes=100)

    load_model(
        model_a, 
        args.checkpoint_a, 
        mapping={
            "downsample": "shortcut",
            "fc": "linear"
        }
    )

    load_model(
        model_b, 
        args.checkpoint_b, 
        mapping={
            "downsample": "shortcut",
            "fc": "linear"
        }
    )

    with torch.no_grad():
        model_b.linear.weight[:50] = 0.0
        model_b.linear.bias[:50] = 0.0

        model_a.linear.weight[50:] = 0.0
        model_a.linear.bias[50:] = 0.0

    model_a.cuda()
    model_b.cuda()
    test_loader = get_cifar100(train=False)
    train_loader = get_cifar100(train=True, bs=256)
    test_loader_a = get_cifar100_split_a(train=False)
    test_loader_b = get_cifar100_split_b(train=False)
    train_loader_a = get_cifar100_split_a(train=True, bs=256)
    train_loader_b = get_cifar100_split_b(train=True, bs=256)

    model_a.eval()
    model_b.eval()

    acc, _ = eval_model(model_a, test_loader_a)
    print(f"model_a on split a dataset acc: {acc * 100:.2f}")


    # acc, _ = eval_model(model_a, test_loader_b)
    # print(f"model_a on split b dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_b, test_loader_b)
    print(f"model_b on split b dataset acc: {acc * 100:.2f}")

    # acc, _ = eval_model(model_b, test_loader_a)
    # print(f"model_b on split a dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_a, test_loader)
    print(f"model_a on joint dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_b, test_loader)
    print(f"model_b on joint dataset acc: {acc * 100:.2f}")

    _,_, fused_model = test_merge(copy.deepcopy(model_a), copy.deepcopy(model_a).state_dict(), copy.deepcopy(model_b), copy.deepcopy(model_b).state_dict(), test_loader, train_loader,  fuse_channel_resnet50_clustering)

    acc, _ = eval_model(fused_model, test_loader_a)
    print(f"fused model on split a dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(fused_model, test_loader_b)
    print(f"fused model on split b dataset acc: {acc * 100:.2f}")

if __name__ == "__main__":
  main()
# CUDA_VISIBLE_DEVICES=3 python resnet50_cifar100_weight_merging_split.py --checkpoint_a ~/project/model_folding_public/weights/resnet50_CIFAR100_split_a.pth --checkpoint_b ~/project/model_folding_public/weights/resnet50_CIFAR100_split_b.pth