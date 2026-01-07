"""
Simplified DETR Loss Function (based on RF-DETR)

Single-file implementation with:
- Hungarian matching
- IoU-aware BCE classification loss (sigmoid)
- L1 bounding box loss
- GIoU loss

Usage:
    criterion = DETRLoss(num_classes=80)
    losses = criterion(pred_logits, pred_boxes, targets)

Where:
    pred_logits: (batch, num_queries, num_classes) - raw logits
    pred_boxes: (batch, num_queries, 4) - cxcywh normalized [0,1]
    targets: list of dicts with COCO-style annotations, each containing:
        - "boxes": (N, 4) in cxcywh normalized [0,1]
        - "labels": (N,) class indices (0-indexed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area
import numpy as np


# ============================================================================
# Box utilities
# ============================================================================


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        x_c - 0.5 * w.clamp(min=0.0),
        y_c - 0.5 * h.clamp(min=0.0),
        x_c + 0.5 * w.clamp(min=0.0),
        y_c + 0.5 * h.clamp(min=0.0),
    ]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    Args:
        boxes1: (N, 4) xyxy
        boxes2: (M, 4) xyxy
    Returns:
        iou: (N, M)
        union: (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    GIoU from https://giou.stanford.edu/

    Args:
        boxes1: (N, 4) xyxy format
        boxes2: (M, 4) xyxy format
    Returns:
        (N, M) pairwise GIoU matrix
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# ============================================================================
# Hungarian Matcher
# ============================================================================


class HungarianMatcher(nn.Module):
    """
    Computes assignment between predictions and ground truth.

    Cost = cost_class * C_class + cost_bbox * C_bbox + cost_giou * C_giou
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        group_detr: int = 1,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.group_detr = group_detr

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits: (batch, num_queries, num_classes)
            pred_boxes: (batch, num_queries, 4) cxcywh normalized
            targets: list of dicts with "boxes" and "labels"
        Returns:
            list of (pred_indices, gt_indices) tuples
        """
        bs, num_queries = pred_logits.shape[:2]

        # Flatten for batched cost computation
        out_prob = pred_logits.flatten(0, 1).sigmoid()
        out_bbox = pred_boxes.flatten(0, 1)

        # Concat all targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Classification cost: focal-weighted
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-pred_logits.flatten(0, 1)))
        )
        pos_cost_class = (
            alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(pred_logits.flatten(0, 1)))
        )
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # L1 bbox cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Combined cost matrix
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        # Handle NaN/Inf (on GPU before transfer)
        if C.numel() > 0:
            is_inf_nan = C.isinf() | C.isnan()
            if is_inf_nan.any():
                max_cost = C[~is_inf_nan].max() if (~is_inf_nan).any() else 0
                C = C.clone()
                C[is_inf_nan] = max_cost * 2

        C = C.cpu()

        # Run Hungarian matching per image (with group DETR support)
        sizes = [len(v["boxes"]) for v in targets]
        g_num_queries = num_queries // self.group_detr
        C_list = C.split(g_num_queries, dim=1)

        indices = []
        for g_i in range(self.group_detr):
            C_g = C_list[g_i]
            indices_g = [
                linear_sum_assignment(c[i].numpy()) for i, c in enumerate(C_g.split(sizes, -1))
            ]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([idx1[0], idx2[0] + g_num_queries * g_i]),
                        np.concatenate([idx1[1], idx2[1]]),
                    )
                    for idx1, idx2 in zip(indices, indices_g)
                ]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


# ============================================================================
# Loss Functions
# ============================================================================


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    """
    Standard focal loss for sigmoid outputs.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# ============================================================================
# Main Loss Class
# ============================================================================


class DETRLoss(nn.Module):
    """
    Simplified DETR loss with:
    - IoU-aware BCE classification (sigmoid, no explicit background class)
    - L1 bbox loss
    - GIoU loss
    """

    def __init__(
        self,
        num_classes: int,
        loss_coef_class: float = 1.0,
        loss_coef_bbox: float = 5.0,
        loss_coef_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        matcher_cost_class: float = 2.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        group_detr: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_coef_class = loss_coef_class
        self.loss_coef_bbox = loss_coef_bbox
        self.loss_coef_giou = loss_coef_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            group_detr=group_detr,
        )

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, pred_logits, pred_boxes, targets, indices, num_boxes):
        """
        IoU-aware BCE classification loss.

        Uses IoU between predicted and GT boxes as soft target weight.
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Get matched boxes
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Compute IoU for matched pairs
        iou_targets = torch.diag(
            box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))[0]
        )
        pos_ious = iou_targets.clone().detach()

        # IoU-aware BCE
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        prob = pred_logits.sigmoid()

        # Initialize weights
        pos_weights = torch.zeros_like(pred_logits)
        neg_weights = prob**gamma

        # Build index for positive samples
        pos_ind = (idx[0], idx[1], target_classes_o)

        # Soft target: blend of predicted prob and IoU
        t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
        t = torch.clamp(t, 0.01).detach()

        pos_weights[pos_ind] = t.to(pos_weights.dtype)
        neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)

        # Numerically stable formulation
        loss_ce = neg_weights * pred_logits - F.logsigmoid(pred_logits) * (
            pos_weights + neg_weights
        )
        loss_ce = loss_ce.sum() / num_boxes

        return {"loss_class": loss_ce}

    def loss_boxes(self, pred_boxes, targets, indices, num_boxes):
        """
        L1 and GIoU loss for bounding boxes.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou.sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def compute_loss(self, outputs, targets):
        """
        Compute DETR losses.

        Args:
            outputs: dict with
                "pred_logits" (#batch x #queries x #classes) and
                "pred_boxes" (#batch x #queries x 4)
            targets: list of dicts, each with:
                - "boxes": (N, 4) cxcywh normalized
                - "labels": (N,) class indices

        Returns:
            dict with loss_class, loss_bbox, loss_giou, and total loss
        """
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]

        device = logits.device

        # Move targets to device
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets
        ]

        # Hungarian matching
        indices = self.matcher(logits, boxes, targets)

        # Move indices to device
        indices = [(i.to(device), j.to(device)) for i, j in indices]

        # Number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute losses
        losses = {}
        losses.update(self.loss_labels(logits, boxes, targets, indices, num_boxes))
        losses.update(self.loss_boxes(boxes, targets, indices, num_boxes))

        # Total weighted loss
        losses["loss"] = (
            self.loss_coef_class * losses["loss_class"]
            + self.loss_coef_bbox * losses["loss_bbox"]
            + self.loss_coef_giou * losses["loss_giou"]
        )

        return losses

    def forward(self, outputs, targets, early_loss_coeffs=[]):
        """Compute loss including auxiliary losses from intermediate layers."""
        losses = {}

        all_outputs = outputs.get("aux_outputs", [])
        final_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        all_outputs.append(final_outputs)

        assert len(all_outputs) == len(early_loss_coeffs)

        for i, (layer_outputs, coeff) in enumerate(zip(all_outputs, early_loss_coeffs)):
            layer_losses = self.compute_loss(layer_outputs, targets)
            for k, v in layer_losses.items():
                losses[f"{k}_{i}"] = v

            loss_contribution = coeff * layer_losses["loss"]

            if "loss" not in losses:
                losses["loss"] = loss_contribution
            else:
                losses["loss"] = losses["loss"] + loss_contribution

        return losses


# ============================================================================
# Test
# ============================================================================


if __name__ == "__main__":
    import time
    from torch.profiler import profile, record_function, ProfilerActivity

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 64
    num_queries = 300
    num_classes = 80
    num_gt_per_image = 20

    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 4, device=device))

    print(pred_logits.shape, pred_boxes.shape)

    targets = [
        {
            "boxes": torch.rand(num_gt_per_image, 4),
            "labels": torch.randint(0, num_classes, (num_gt_per_image,)),
        }
        for _ in range(batch_size)
    ]

    criterion = DETRLoss(num_classes=num_classes)

    # Warmup
    for _ in range(3):
        losses = criterion(pred_logits, pred_boxes, targets)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            losses = criterion(pred_logits, pred_boxes, targets)
            torch.cuda.synchronize()

    # Print table sorted by CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Export for Chrome trace viewer (open chrome://tracing and load this file)
    prof.export_chrome_trace("loss_trace.json")
    print("\nTrace saved to loss_trace.json - open in chrome://tracing")
