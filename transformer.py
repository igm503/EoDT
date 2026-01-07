import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import numpy as np


class DetectionTransformer(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        num_classes: int = 80,
        num_queries: int = 300,
        query_layers: int = 4,
        use_separate_heads: bool = False,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.preprocess = AutoImageProcessor.from_pretrained(backbone_name)
        self.num_classes = num_classes
        self.num_queries = num_queries

        hidden_dim = self.backbone.config.hidden_size

        num_heads = query_layers if use_separate_heads else 1

        self.bbox_mlp = nn.ModuleList()
        self.cls_mlp = nn.ModuleList()

        for _ in range(num_heads):
            self.bbox_mlp.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 4),
                )
            )

            self.cls_mlp.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_classes),
                )
            )

            # Initialize bbox head for small outputs
            nn.init.constant_(self.bbox_mlp[-1][-1].weight, 0)
            nn.init.constant_(self.bbox_mlp[-1][-1].bias, 0)

            # Initialize cls head with prior for focal loss
            prior_prob = 0.01
            bias_value = -np.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_mlp[-1][-1].bias, bias_value)

        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Find encoder layers
        if hasattr(self.backbone, "encoder"):
            self.encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layer"):
            self.encoder_layers = self.backbone.layer
        else:
            raise ValueError("Cannot find encoder layers in backbone")

        self.first_query_layer = max(0, len(self.encoder_layers) - query_layers)


    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        batch_size = x.shape[0]
        pos = self.backbone.rope_embeddings(x)
        x = self.backbone.embeddings(x)

        intermediate_queries = []

        for i, layer in enumerate(self.backbone.layer):
            if i == self.first_query_layer:
                query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                x = torch.cat([query_tokens, x], dim=1)
            x = layer(x, position_embeddings=pos)

            if return_intermediate and i >= self.first_query_layer:
                intermediate_queries.append(x[:, : self.num_queries, :])

        # Final norm
        if hasattr(self.backbone, "layernorm"):
            x = self.backbone.layernorm(x)
        elif hasattr(self.backbone, "norm"):
            x = self.backbone.norm(x)
        elif hasattr(self.backbone.encoder, "layer_norm"):
            x = self.backbone.encoder.layer_norm(x)

        final_queries = x[:, : self.num_queries, :]

        bboxes = torch.sigmoid(self.bbox_mlp[-1](final_queries))
        scores = self.cls_mlp[-1](final_queries)

        output = {"pred_boxes": bboxes, "pred_logits": scores}

        if return_intermediate:
            aux_outputs = []
            for i, q in enumerate(intermediate_queries[:-1]):  # exclude last (it becomes the final)
                if len(self.bbox_mlp) == 1:  # single head
                    i = 0
                aux_outputs.append(
                    {
                        "pred_boxes": torch.sigmoid(self.bbox_mlp[i](q)),
                        "pred_logits": self.cls_mlp[i](q),
                    }
                )
            output["aux_outputs"] = aux_outputs

        return output


if __name__ == "__main__":
    import cv2
    import torch.nn.functional as F
    from constants import COCO_CLASSES

    weights_path = "runs/detr_dinov3_small_640_1/checkpoints/checkpoint_best.pt"
    cv_image = cv2.imread("test.jpg")
    orig_h, orig_w, _ = cv_image.shape
    image = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
    image = F.interpolate(
        image.unsqueeze(0), size=(640, 640), mode="bilinear", align_corners=False
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    image = image.to("cuda")

    # Load checkpoint
    checkpoint = torch.load(weights_path, weights_only=False)
    # Strip the "_orig_mod." prefix from all keys
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # Load into uncompiled model
    cfg = checkpoint["config"]
    model = DetectionTransformer(
        backbone_name=cfg["backbone"],
        num_classes=80,
        num_queries=cfg["num_queries"],
        query_layers=cfg["query_layers"],
    )
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    model.eval()

    print(image.shape)
    with torch.no_grad():
        for i in tqdm(range(10)):
            model(image.unsqueeze(0))
        results = model(image.unsqueeze(0))

    for box, scores in zip(
        results["pred_boxes"][0].cpu().detach().numpy(),
        results["pred_logits"][0].cpu().detach(),
    ):
        scores = torch.sigmoid(scores).numpy()
        class_id = np.argmax(scores)
        score = scores[class_id]
        if score < 0.4:
            continue

        cx = box[0] * orig_w
        cy = box[1] * orig_h
        w = box[2] * orig_w
        h = box[3] * orig_h
        x1 = int(max(0, cx - w / 2))
        y1 = int(max(0, cy - h / 2))
        x2 = int(min(orig_w, cx + w / 2))
        y2 = int(min(orig_h, cy + h / 2))

        # Draw bounding box
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create label text
        label = f"{COCO_CLASSES[class_id]}: {score:.2f}"

        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw filled rectangle behind text
        cv2.rectangle(cv_image, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), (0, 255, 0), -1)

        # Draw text
        cv2.putText(cv_image, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)

    cv2.imwrite("test_output.jpg", cv_image)
