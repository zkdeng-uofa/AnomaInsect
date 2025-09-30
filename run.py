import os
import sys
from pathlib import Path

### Add the parent directory of 'models' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, parent_dir)

import json
import datetime
import albumentations as A
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import warnings
import argparse
import time
import random

from PIL import Image
from albumentations.pytorch import ToTensorV2
from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore
import torch.nn.functional as F
from anomalib.data import Folder
from anomalib.loggers import AnomalibWandbLogger

from pytorch_lightning import seed_everything, Trainer
# 2. Seed python random, numpy, torch (CPU & GPU)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 3. cuDNN settings (for deterministic behavior)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 4. If using PyTorch >= something, enforce deterministic algorithms
# (this may make some ops complain if no deterministic variant exists)
try:
    torch.use_deterministic_algorithms(True)
except AttributeError:
    pass

# 5. In PyTorch Lightning, use seed_everything, with worker seeding
seed_everything(seed, workers=True)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_fields = [
            'train_batch_size', 
            'eval_batch_size', 
            'num_workers', 
            'coreset_sampling_ratio', 
            'num_neighbors', 
            'max_epochs', 
            'dataset_root', 
            'image_size',
            'model',
            'layers',
            'training_dir',
            'backbone'
        ]
        
        # Optional memory optimization fields
        optional_fields = ['memory_efficient', 'chunk_size', 'use_mixed_precision']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config file")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection model")
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    parser.add_argument("--train_batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.1, help="Coreset sampling ratio")
    parser.add_argument("--num_neighbors", type=int, default=12, help="Number of neighbors")
    parser.add_argument("--max_epochs", type=int, default=1, help="Max epochs")
    parser.add_argument("--dataset_root", type=str, help="Dataset root")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--model", type=str, default="PatchCore", help="Model")
    parser.add_argument("--layers", type=json.loads, default=["layer2", "layer3"], help="Layers")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone")
    parser.add_argument("--training_dir", type=str, help="Training directory")

    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if value is not None and key != 'config':
                config[key] = value
        for key, value in config.items():
            setattr(args, key, value)
    else:
        required_args = [
            'training_dir']
        for arg in required_args:
            if getattr(args, arg) is None:
                parser.error(f"--{arg} is required when not using --config")
    
    return args


def albu_adapter(aug):
    """Wrap an Albumentations Compose so it accepts (image) and returns a Tensor.
       Converts PIL/Tensor to NumPy HxWxC uint8 before calling Albumentations.
    """
    def _call(image):
        # 1) Normalize input type to NumPy HxWxC
        if isinstance(image, np.ndarray):
            img = image
            # If CHW, convert to HWC
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
        elif isinstance(image, Image.Image):
            img = np.array(image)  # PIL -> HWC uint8
        elif isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            # Assume CHW; convert to HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            # If float in [0,1], convert to uint8 0..255 for Albumentations Normalize defaults
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1) * 255.0
                arr = arr.astype(np.uint8)
            img = arr
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected numpy, PIL, or torch.Tensor.")

        if img.ndim == 2:
            img = img[..., None]

        # 2) Call Albumentations with named argument
        out = aug(image=img)
        
        return out["image"]
    return _call

def define_transforms(image_size):
    train_aug = A.Compose([
        A.Resize(image_size, image_size),
        # Optional robustifying augs for field images:
        # A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # expects uint8 input
        ToTensorV2(),
    ])

    eval_aug = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_tf = albu_adapter(train_aug)
    eval_tf  = albu_adapter(eval_aug)

    return train_tf, eval_tf

def wandb_init(args):
    wandb_logger = AnomalibWandbLogger(
        project="spider-anomaly-detection",
        name="patchcore-spider-experiment",
        save_dir="./logs"
    )

    wandb.init(
        project="spider-anomaly-detection",
        name="patchcore-spider-experiment",
        tags=["patchcore", "spiders", "anomaly-detection", "resnet18"],
        notes="Spider anomaly detection using PatchCore with detailed evaluation",
        config={
            "model": "PatchCore",
            "backbone": args.backbone,
            "layers": args.layers,
            "coreset_sampling_ratio": args.coreset_sampling_ratio,
            "num_neighbors": args.num_neighbors,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_epochs": 1,
            "dataset_root": args.dataset_root,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
        }
    )

    print("Weights & Biases logger initialized!")
    print(f"Project: {wandb.run.project}")
    print(f"Run name: {wandb.run.name}")
    print(f"Run URL: {wandb.run.url}")
    return wandb_logger

def post_process_results(post_processor):
    threshold_value = None
    threshold_attrs = ['threshold', 'threshold_', 'image_threshold', 'pixel_threshold']
    for attr in threshold_attrs:
        if hasattr(post_processor, attr):
            threshold_value = getattr(post_processor, attr)
            break
    
    normalized_threshold_value = None
    normalized_threshold_attrs = ['normalized_image_threshold']
    for attr in normalized_threshold_attrs:
        if hasattr(post_processor, attr):
            normalized_threshold_value = getattr(post_processor, attr)
            break

    return threshold_value, normalized_threshold_value

def main():
    args = parse_args()
    start_time = time.time()
    warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

    train_tf, eval_tf = define_transforms(args.image_size)
    wandb_logger = wandb_init(args)

    dm = Folder(
        name="spider_anomaly_detection",
        root=args.training_dir,
        normal_dir="normal",
        abnormal_dir="abnormal",
        train_augmentations=train_tf,
        val_augmentations=eval_tf,
        test_augmentations=eval_tf,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,
    )

    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        visualizer=False,
    )

    engine = Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    print("🚀 Starting training with wandb logging...")
    engine.fit(datamodule=dm, model=model)

    print("🧪 Running evaluation with wandb logging...")
    test_results = engine.test(datamodule=dm, model=model)

    threshold_value, normalized_threshold_value = post_process_results(model.post_processor)

    if threshold_value is not None:
        print(f"Threshold Value       : {threshold_value:.6f}")
    if normalized_threshold_value is not None:
        print(f"Normalized Threshold Value       : {normalized_threshold_value:.6f}")
    else:
        print(f"Threshold Value       : Not accessible")
    print(f"Threshold Type        : {type(model.post_processor).__name__}")

    if hasattr(model.post_processor, 'normalization_method'):
        print(f"Normalization Method  : {model.post_processor.normalization_method}")

    print(f"\n💡 MODEL CONFIGURATION:")
    print("=" * 50)

    # Get model configuration for logging
    model_config = {}
    if hasattr(model, 'hparams'):
        model_config = dict(model.hparams)
        print(model_config)
    else:
        print("Model attributes (first 10):")
        attrs = [attr for attr in dir(model) if not attr.startswith('_')][:10]
        print(attrs)
        print("Standard configuration attributes not accessible")
        # Fallback to args config
        model_config = {
            "backbone": args.backbone,
            "layers": args.layers,
            "coreset_sampling_ratio": args.coreset_sampling_ratio,
            "num_neighbors": args.num_neighbors,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "image_size": args.image_size,
        }

    print(f"Time taken           : {time.time() - start_time:.2f} seconds")

    basic_metrics = {
        "test/image_AUROC": test_results[0].get("image_AUROC", 0),
        "test/image_F1Score": test_results[0].get("image_F1Score", 0),
        "test/threshold_value": threshold_value,
        "test/threshold_type": type(model.post_processor).__name__,
        "test/model_configuration": model_config,
        "training/duration_seconds": time.time() - start_time,
    }

    wandb.log(basic_metrics)

    print(f"\n📈 BASIC TEST RESULTS:")
    for metric, value in test_results[0].items():
        print(f"{metric:20s}: {value:.6f}")

    print(f"\n⏱️  Training completed in: {time.time() - start_time:.2f} seconds")
    print(f"🌐 View results at: {wandb.run.url}")

if __name__ == "__main__":
    main()
    

