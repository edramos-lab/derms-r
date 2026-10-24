"""Canonical DERMS-R constants. Do not change these defaults except for explicit sensitivity analysis."""
RATIOS = (0.10, 0.25, 0.50, 0.75, 1.00)
ALPHA = 2.0
BETA = 0.5
GAMMA = 0.5
DELTA = 0.5
P0 = 1_000_000.0
SIGMA0 = 0.001
FOLDS = 5
REGISTRY = {
    'resnet50': 'resnet50.a1_in1k',
    'efficientnet_b0': 'efficientnet_b0.ra_in1k',
    'efficientnetv2_s': 'tf_efficientnetv2_s.in1k',
    'convnextv2_tiny': 'convnextv2_tiny.fcmae_ft_in1k',
    'xception41': 'xception41.tf_in1k',
    'vit_base': 'vit_base_patch16_224.augreg_in1k',
    'deit3_base': 'deit3_base_patch16_224.fb_in1k',
    'swin_tiny': 'swin_tiny_patch4_window7_224.ms_in1k',
    'maxvit_tiny': 'maxvit_tiny_tf_224.in1k',
    'nextvit_base': 'nextvit_base.bd_in1k',
    'efficientvit_b0': 'efficientvit_b0.r224_in1k',
    'mobilenetv4': 'mobilenetv4_conv_small.e2400_r224_in1k',
}
DATASETS = ('cifar10', 'cifar100', 'flowers102', 'food101', 'all', 'bloodmnist')
# Dataset names backed by this repo's own ImageFolder layout (development/test
# folders under datasets/<name>), as opposed to a torchvision-provided dataset.
IMAGEFOLDER_DATASETS = ('all', 'bloodmnist')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SCHEMA = 5
