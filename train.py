import shutil
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import numpy as np

import wandb
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from loss import DETRLoss
from transformer import DetectionTransformer
from constants import COCO_ID_TO_IDX, IDX_TO_COCO_ID, NUM_CLASSES


# ============================================================================
# Dataset
# ============================================================================


class COCODetectionDataset(Dataset):
    """
    COCO Detection dataset for fiftyone directory structure.

    Expected structure:
        {root}/train/data/*.jpg
        {root}/validation/data/*.jpg
        {root}/raw/instances_train2017.json
        {root}/raw/instances_val2017.json
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: int = 640,
    ):
        self.root = Path(root)
        self.split = split
        self.resolution = resolution

        # Paths
        if split == "train":
            self.img_dir = self.root / "train" / "data"
            ann_file = self.root / "raw" / "instances_train2017.json"
        else:
            self.img_dir = self.root / "validation" / "data"
            ann_file = self.root / "raw" / "instances_val2017.json"

        # Load COCO annotations
        self.coco = COCO(str(ann_file))

        # Get image IDs that have annotations
        self.img_ids = list(self.coco.imgToAnns.keys())

        # Filter to images that exist on disk
        existing_ids = []
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = self.img_dir / img_info["file_name"]
            if img_path.exists():
                existing_ids.append(img_id)
        self.img_ids = existing_ids

        print(f"[{split}] Loaded {len(self.img_ids)} images with annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to square
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)

        # Convert to tensor and normalize (ImageNet stats, same as DINOv2)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Convert to cxcywh normalized [0, 1]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            boxes.append([cx, cy, nw, nh])

            coco_cat_id = ann["category_id"]
            if coco_cat_id in COCO_ID_TO_IDX:
                labels.append(COCO_ID_TO_IDX[coco_cat_id])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return img, {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "orig_size": (orig_w, orig_h),
        }


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate_coco(model, dataloader, device, score_threshold=0.01):
    """Evaluate using pycocotools."""
    model.eval()

    coco_gt = dataloader.dataset.coco
    results = []

    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        outputs = model(images)
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]

        for i, target in enumerate(targets):
            image_id = target["image_id"]
            orig_w, orig_h = target["orig_size"]

            boxes = pred_boxes[i]
            logits = pred_logits[i]

            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)

            keep = scores > score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(boxes) > 0:
                # cxcywh normalized -> xywh pixels
                boxes_xywh = torch.zeros_like(boxes)
                boxes_xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w
                boxes_xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h
                boxes_xywh[:, 2] = boxes[:, 2] * orig_w
                boxes_xywh[:, 3] = boxes[:, 3] * orig_h

                for j in range(len(boxes)):
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": IDX_TO_COCO_ID[labels[j].item()],
                            "bbox": boxes_xywh[j].cpu().tolist(),
                            "score": scores[j].item(),
                        }
                    )

    if len(results) == 0:
        print("No predictions above threshold!")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
    }

    return metrics


# ============================================================================
# Training
# ============================================================================


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device,
    scaler,
    use_amp,
    epoch,
    early_loss_coeffs=[],
):
    model.train()

    total_loss = 0.0
    total_losses = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for images, targets in pbar:
        optimizer.zero_grad()

        with autocast(device_type=str(device), enabled=use_amp):
            images = images.to(device)

            outputs = model(images, return_intermediate=bool(early_loss_coeffs))
            losses = criterion.forward(outputs, targets, early_loss_coeffs)
            loss = losses["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        total_loss += losses["loss"].item()
        num_batches += 1

        if early_loss_coeffs:
            final_loss_idx = len(early_loss_coeffs) - 1
        else:
            final_loss_idx = 0

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{losses['loss'].item():.4f}",  # total loss
                "loss_final": f"{losses[f'loss_{final_loss_idx}'].item():.4f}",
                "cls": f"{losses[f'loss_class_{final_loss_idx}'].item():.4f}",
                "bbox": f"{losses[f'loss_bbox_{final_loss_idx}'].item():.4f}",
                "giou": f"{losses[f'loss_giou_{final_loss_idx}'].item():.4f}",
            }
        )

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses


@torch.no_grad()
def validate(model, criterion, dataloader, device, use_amp, early_loss_coeffs=[]):
    model.eval()

    total_loss = 0.0
    total_losses = {}
    num_batches = 0

    for images, targets in tqdm(dataloader, desc="Validating", leave=False):
        with autocast(device_type=str(device), enabled=use_amp):
            images = images.to(device)
            outputs = model(images, return_intermediate=bool(early_loss_coeffs))
            losses = criterion.forward(outputs, targets, early_loss_coeffs)

        total_loss += losses["loss"].item()

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()

        num_batches += 1

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses


# ============================================================================
# Main
# ============================================================================


def create_run_dir(base_dir: str, run_name: str) -> Path:
    """Create run directory, handling existing names with suffixes."""
    base_path = Path(base_dir)
    run_path = base_path / run_name

    if not run_path.exists():
        run_path.mkdir(parents=True)
        return run_path

    # Find available suffix
    i = 1
    while True:
        run_path = base_path / f"{run_name}_{i}"
        if not run_path.exists():
            run_path.mkdir(parents=True)
            return run_path
        i += 1


def log_message(log_file, message: str):
    """Write message to log file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        for key in cfg:
            try:
                cfg[key] = int(cfg[key])
            except:
                try:
                    cfg[key] = float(cfg[key])
                except:
                    continue

    # Create run directory
    run_dir = create_run_dir(cfg["output_dir"], cfg["run_name"])
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()

    # Copy config
    shutil.copy(args.config, run_dir / "config.yaml")

    # Setup logging
    log_file = run_dir / "log.txt"
    log_message(log_file, f"Starting training run: {run_dir}")
    log_message(log_file, f"Config: {cfg}")

    # Device
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    log_message(log_file, f"Using device: {device}")

    # early losses
    if cfg.get("early_loss", False):
        early_loss_coeffs = cfg.get("early_loss_coefs", [1.0] * cfg["query_layers"])
        assert len(early_loss_coeffs) == cfg["query_layers"]
        use_separate_heads = cfg.get("use_separate_heads", False)
    else:
        early_loss_coeffs = []
        use_separate_heads = False

    # Datasets
    train_dataset = COCODetectionDataset(
        root=cfg["coco_dir"],
        split="train",
        resolution=cfg["resolution"],
    )
    val_dataset = COCODetectionDataset(
        root=cfg["coco_dir"],
        split="val",
        resolution=cfg["resolution"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    log_message(
        log_file,
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}",
    )

    # Model
    model = DetectionTransformer(
        backbone_name=cfg["backbone"],
        num_classes=NUM_CLASSES,
        num_queries=cfg["num_queries"],
        query_layers=cfg["query_layers"],
        use_separate_heads=use_separate_heads,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(log_file, f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Loss
    criterion = DETRLoss(
        num_classes=NUM_CLASSES,
        loss_coef_class=cfg.get("loss_coef_class", 2.0),
        loss_coef_bbox=cfg.get("loss_coef_bbox", 5.0),
        loss_coef_giou=cfg.get("loss_coef_giou", 2.0),
    )

    # Optimizer
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": cfg["lr_backbone"],
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": cfg["lr"],
        },
    ]
    optimizer = AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)

    # Mixed precision
    use_amp = bool(cfg.get("use_amp", True))
    scaler = GradScaler(enabled=use_amp)

    # Training loop
    best_ap = 0.0

    model = torch.compile(model, mode="max-autotune")

    # Initialize wandb
    wandb.init(
        project=cfg.get("wandb_project", "detr-training"),
        name=cfg["run_name"],
        config=cfg,
        dir=str(run_dir),
    )

    for epoch in range(1, cfg["epochs"] + 1):
        log_message(log_file, f"\n{'=' * 60}")
        log_message(log_file, f"Epoch {epoch}/{cfg['epochs']}")

        # Train
        train_losses = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            epoch,
            early_loss_coeffs,
        )

        scheduler.step()

        if early_loss_coeffs:
            final_loss_idx = len(early_loss_coeffs) - 1
        else:
            final_loss_idx = 0

        log_message(
            log_file,
            f"Train - loss: {train_losses['loss']:.4f}, "
            f"cls: {train_losses[f'loss_class_{final_loss_idx}']:.4f}, "
            f"bbox: {train_losses[f'loss_bbox_{final_loss_idx}']:.4f}, "
            f"giou: {train_losses[f'loss_giou_{final_loss_idx}']:.4f}",
        )

        # Validate
        val_losses = validate(model, criterion, val_loader, device, use_amp, early_loss_coeffs)
        log_message(
            log_file,
            f"Val   - loss: {val_losses['loss']:.4f}, "
            f"cls: {val_losses[f'loss_class_{final_loss_idx}']:.4f}, "
            f"bbox: {val_losses[f'loss_bbox_{final_loss_idx}']:.4f}, "
            f"giou: {val_losses[f'loss_giou_{final_loss_idx}']:.4f}",
        )

        # COCO evaluation
        metrics = evaluate_coco(model, val_loader, device)
        log_message(
            log_file,
            f"Val   - AP: {metrics['AP']:.4f}, "
            f"AP50: {metrics['AP50']:.4f}, "
            f"AP75: {metrics['AP75']:.4f}",
        )

        # Log to wandb
        wandb_update = {
            "epoch": epoch,
            "val/AP": metrics["AP"],
            "val/AP50": metrics["AP50"],
            "val/AP75": metrics["AP75"],
            "val/AP_small": metrics["AP_small"],
            "val/AP_medium": metrics["AP_medium"],
            "val/AP_large": metrics["AP_large"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        for k, v in train_losses.items():
            if k.endswith(f"_{final_loss_idx}"):
                wandb_update[f"train/{k[:-2]}"] = v
            else:
                wandb_update[f"train/{k}"] = v

        for k, v in val_losses.items():
            if k.endswith(f"_{final_loss_idx}"):
                wandb_update[f"val/{k[:-2]}"] = v
            else:
                wandb_update[f"val/{k}"] = v

        wandb.log(wandb_update)

        if hasattr(model, "_orig_mod"):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
            "config": cfg,
        }

        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

        # Save best model
        if metrics["AP"] > best_ap:
            best_ap = metrics["AP"]
            torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pt")
            log_message(log_file, f"New best AP: {best_ap:.4f}")

        # Keep only last N checkpoints to save space
        keep_last = cfg.get("keep_last_checkpoints", 5)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        for ckpt in checkpoints[:-keep_last]:
            ckpt.unlink()

    log_message(log_file, f"\nTraining complete! Best AP: {best_ap:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
