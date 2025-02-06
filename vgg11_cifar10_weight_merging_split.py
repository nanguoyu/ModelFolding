from model.vgg import fuse_channel_vgg11_clustering
from torchvision.models.vgg import make_layers, VGG
from utils.datasets import get_cifar10
from utils.utils import fuse_bnorms_vgg, eval_model
from utils.datasets import get_cifar10, get_cifar10_split_a, get_cifar10_split_b


import argparse
import torch
import wandb
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

    return acc, param / origin_param



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", type=str)
    parser.add_argument("--checkpoint_b", type=str)
    parser.add_argument("--width", type=int, default=1)

    args = parser.parse_args()

    # Create model configurations
    vgg11_cfg  = [args.width * 64, 'M', args.width * 128, 'M', args.width * 256, args.width * 256, 'M', args.width * 512, args.width * 512, 'M', args.width * 512, 512, 'M']
    features_a   = make_layers(vgg11_cfg, batch_norm=True)
    features_b   = make_layers(vgg11_cfg, batch_norm=True)
    
    # Create two identical model architectures
    model_a    = VGG(features=features_a, num_classes=10)
    model_b    = VGG(features=features_b, num_classes=10)
    
    # Load checkpoints with error handling
    try:
        checkpoint_a = torch.load(args.checkpoint_a, map_location="cpu")
        checkpoint_b = torch.load(args.checkpoint_b, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoints: {e}")

    # Load state dictionaries with strict checking
    model_a.load_state_dict(checkpoint_a, strict=True)
    model_b.load_state_dict(checkpoint_b, strict=True)
    model_a.cuda()
    model_b.cuda()
    
    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)
    test_loader_a = get_cifar10_split_a(train=False)
    test_loader_b = get_cifar10_split_b(train=False)
    train_loader_a = get_cifar10_split_a(train=True, bs=256)
    train_loader_b = get_cifar10_split_b(train=True, bs=256)

    with torch.no_grad():
        model_b.classifier[6].weight[:5] = 0.0
        model_b.classifier[6].bias[:5] = 0.0

        model_a.classifier[6].weight[5:] = 0.0
        model_a.classifier[6].bias[5:] = 0.0

    model_a.eval()
    model_b.eval()
    print(model_a)

    acc, _ = eval_model(model_a, test_loader_a)
    print(f"model_a on split a dataset acc: {acc * 100:.2f}")


    acc, _ = eval_model(model_a, test_loader_b)
    print(f"model_a on split b dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_b, test_loader_b)
    print(f"model_b on split b dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_b, test_loader_a)
    print(f"model_b on split a dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_a, test_loader)
    print(f"model_a on joint dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(model_b, test_loader)
    print(f"model_b on joint dataset acc: {acc * 100:.2f}")

    _,_, fused_model = test_merge(copy.deepcopy(model_a), copy.deepcopy(model_a).state_dict(), copy.deepcopy(model_b), copy.deepcopy(model_b).state_dict(), test_loader, train_loader,  fuse_channel_vgg11_clustering)

    acc, _ = eval_model(fused_model, test_loader_a)
    print(f"fused model on split a dataset acc: {acc * 100:.2f}")

    acc, _ = eval_model(fused_model, test_loader_b)
    print(f"fused model on split b dataset acc: {acc * 100:.2f}")


if __name__ == "__main__":
  main()
# CUDA_VISIBLE_DEVICES=3 python vgg11_cifar10_weight_merging_split.py --checkpoint_a ~/project/model_folding_public/weights/vgg11_bn_CIFAR10_split_a.pth --checkpoint_b ~/project/model_folding_public/weights/vgg11_bn_CIFAR10_split_b.pth