import json
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from dino_finetune import (
    DINOEncoderLoRA,
    get_dataloader,
    visualize_overlay,
    compute_iou_metric,
)


def validate_epoch(
    dino_lora: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    metrics: dict,
) -> None:
    val_loss = 0.0
    val_iou = 0.0

    dino_lora.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().cuda()
            masks = masks.long().cuda()

            logits = dino_lora(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            y_hat = torch.sigmoid(logits)
            iou_score = compute_iou_metric(y_hat, masks, ignore_index=255)
            val_iou += iou_score.item()

    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["val_iou"].append(val_iou / len(val_loader))


def finetune_dino(config: argparse.Namespace, encoder: nn.Module):
    dino_lora = DINOEncoderLoRA(
        encoder=encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
        use_fpn=config.use_fpn,
    ).cuda()

    if config.lora_weights:
        dino_lora.load_parameters(config.lora_weights)

    train_loader, val_loader = get_dataloader(
        config.dataset, 
        img_dim=config.img_dim, 
        batch_size=config.batch_size,
        dataset_root=getattr(config, 'dataset_root', None),
        n_classes=config.n_classes if config.dataset == "openfake" else None,
    )

    # Finetuning for segmentation
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
    optimizer = optim.AdamW(dino_lora.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler start warm-up with steady incline, at config.warmup_epochs, start cosine annealing    
    warmup_sched  = LambdaLR(optimizer, lambda epoch: min(1.0, (epoch + 1) / config.warmup_epochs))
    cos_sched = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=config.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cos_sched], milestones=[config.warmup_epochs])
    
    # Log training and validation metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
    }

    print(f"Training started for {config.epochs} epochs")
    for epoch in range(config.epochs):
        dino_lora.train()
        train_loss = 0.0
        num_batches = 0

        logging.info(f"Epoch {epoch+1}/{config.epochs} - Training started")
        
        for batch_idx, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}"):
            images = images.float().cuda()
            masks = masks.long().cuda()
            optimizer.zero_grad()

            logits = dino_lora(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # 매 100 배치마다 진행 상황 출력
            if (batch_idx + 1) % 100 == 0:
                avg_loss = train_loss / num_batches
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch {epoch+1}/{config.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                    f"Train Loss: {avg_loss:.4f} - LR: {current_lr:.6f}"
                )
    
        scheduler.step()
        
        # 매 epoch마다 평균 train loss 계산
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        metrics["train_loss"].append(avg_train_loss)

        # Validation은 매 epoch마다 실행
        validate_epoch(dino_lora, val_loader, criterion, metrics)
        
        # 매 epoch마다 로그 출력
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {metrics['val_loss'][-1]:.4f} - "
            f"Val IoU: {metrics['val_iou'][-1]:.4f} - "
            f"LR: {current_lr:.6f}"
        )
        
        # 5 epoch마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            dino_lora.save_parameters(f"output/{config.exp_name}_e{epoch+1}.pt")
            logging.info(f"Checkpoint saved: output/{config.exp_name}_e{epoch+1}.pt")

        if config.debug:
            # Visualize some of the batch and write to files when debugging
            y_hat = torch.sigmoid(logits)
            visualize_overlay(
                images, y_hat, config.n_classes, filename=f"viz_{epoch+1}"
            )

    # Log metrics & save model the final values
    # Saves only loRA parameters and classifer
    dino_lora.save_parameters(f"output/{config.exp_name}.pt")
    with open(f"output/{config.exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lora",
        help="Experiment name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug by visualizing some of the outputs to file for a sanity check",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="loRA rank parameter r",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="large",
        help="DINOv2, DINOv3 backbone parameter [small, base, large, giant]",
    )
    parser.add_argument(
        "--dino_type",
        type=str,
        default="dinov3",
        help="Either [dinov2, dinov3], defaults to DINOv3",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use Low-Rank Adaptation (LoRA) to finetune",
    )
    parser.add_argument(
        "--use_fpn",
        action="store_true",
        help="Use the FPN decoder for finetuning",
    )
    parser.add_argument(
        "--img_dim",
        type=int,
        nargs=2,
        default=(490, 490),
        help="Image dimensions (height width)",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Load the LoRA weights from file location",
    )

    # Training parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="ade20k",
        help="The dataset to finetune on, either `voc` or `ade20k`",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=20,
        help="Number of epochs of the training epochs for which we do a warm-up.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=3e-5,
        help="lowest learning rate for the scheduler"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="The weight decay parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory for OpenFake dataset (parquet files location). Required for openfake dataset.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to local checkpoint file (.pth) for DINOv3. If provided, will use this instead of downloading.",
    )
    parser.add_argument(
        "--dinov3_repo_dir",
        type=str,
        default=None,
        help="Local directory path to dinov3 repository. Defaults to torch hub cache if not provided.",
    )
    config = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Dataset configuration
    dataset_classes = {"voc": 21, "ade20k": 150, "openfake": 2}
    if config.dataset not in dataset_classes:
        parser.error(f"Unknown dataset: {config.dataset}. Choose from {list(dataset_classes.keys())}")
    config.n_classes = dataset_classes[config.dataset]
    
    # OpenFake 데이터셋 경로 설정
    if config.dataset == "openfake":
        if config.dataset_root is None:
            config.dataset_root = r"D:\dataset\OpenFake\data"

    # Model configuration
    config.patch_size = 16 if config.dino_type == "dinov3" else 14
    backbones = {
        "small": f"{config.dino_type}_vits{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "base": f"{config.dino_type}_vitb{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "large": f"{config.dino_type}_vitl{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "giant": f"{config.dino_type}_vitg{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "huge": f"{config.dino_type}_vith{config.patch_size}{'plus' if config.dino_type == 'dinov3' else ''}{'_reg' if config.dino_type == 'dinov2' else ''}",
    }

    # DINOv3 로컬 가중치 사용
    if config.dino_type == "dinov3" and config.checkpoint_path:
        import os
        if not os.path.exists(config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint_path}")
        
        # REPO_DIR 설정: 사용자 제공 또는 torch hub 캐시
        if config.dinov3_repo_dir:
            repo_dir = config.dinov3_repo_dir
        else:
            # torch hub 캐시 경로 사용
            repo_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
            if not os.path.exists(repo_dir):
                # 대안: 프로젝트 내 dinov3 디렉토리
                repo_dir = os.path.join(os.path.dirname(__file__), "dinov3")
                if not os.path.exists(repo_dir):
                    raise RuntimeError(
                        f"DINOv3 repository not found. Please either:\n"
                        f"1. Provide --dinov3_repo_dir with path to dinov3 repository\n"
                        f"2. Or ensure torch hub has cached dinov3 at: {os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov3_main')}"
                    )
        
        logging.info(f"Loading DINOv3 from local checkpoint: {config.checkpoint_path}")
        logging.info(f"Using repository directory: {repo_dir}")
        
        # 모델 이름 매핑 (backbone 이름을 dinov3 모델 이름으로 변환)
        model_name_map = {
            "small": "dinov3_vits16",
            "base": "dinov3_vitb16",
            "large": "dinov3_vitl16",
            "giant": "dinov3_vitg16",
            "huge": "dinov3_vith16plus",
        }
        model_name = model_name_map.get(config.size, f"dinov3_vit{config.size[0]}16")
        
        encoder = torch.hub.load(
            repo_or_dir=repo_dir,
            model=model_name,
            source="local",
            weights=config.checkpoint_path,
        ).cuda()
    else:
        # 기존 방식: torch.hub에서 다운로드 또는 DINOv2
        encoder = torch.hub.load(
            repo_or_dir=f"facebookresearch/{config.dino_type}",
            model=backbones[config.size],
        ).cuda()
    config.emb_dim = encoder.num_features

    if config.img_dim[0] % config.patch_size != 0 or config.img_dim[1] % config.patch_size != 0:
        logging.info(f"The image size ({config.img_dim}) should be divisible "
            f"by the patch size {config.patch_size}.")
        # subtract the difference from image size and set a new size.
        config.img_dim = (config.img_dim[0] - config.img_dim[0] % config.patch_size,
                          config.img_dim[1] - config.img_dim[1] % config.patch_size)
        logging.info(f"The image size is lowered to ({config.img_dim}).")
    finetune_dino(config, encoder)
