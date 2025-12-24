import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor


class DetectionTransformer(nn.Module):
    def __init__(
        self,
        backbone_name="google/vit-base-patch16-224",
        num_classes=2,
        query_layers=4,
        num_queries=100,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.preprocess = AutoImageProcessor.from_pretrained(backbone_name)
        self.num_classes = num_classes

        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.backbone.config.hidden_size, 4),
        )

        self.cls_mlp = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.backbone.config.hidden_size, self.num_classes),
        )

        self.query_tokens = nn.Parameter(torch.randn(num_queries, self.backbone.config.hidden_size))
        self.first_query_layer = max(0, len(self.backbone.layer) - query_layers)
        self.num_queries = num_queries

    def forward(self, x):
        x = self.preprocess(x)
        x = torch.stack(x.pixel_values)
        batch_size = x.shape[0]
        pos = self.backbone.rope_embeddings(x)
        x = self.backbone.embeddings(x)

        for i, layer in enumerate(self.backbone.layer):
            if i == self.first_query_layer:
                query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                x = torch.cat([query_tokens, x], dim=1)
            x = layer(x, position_embeddings=pos)

        x = self.backbone.norm(x)

        final_queries = x[:, :self.num_queries, :]

        bboxes = self.bbox_mlp(final_queries)
        scores = self.cls_mlp(final_queries)

        return bboxes, scores


if __name__ == "__main__":
    image = torch.randn(1, 3, 640, 640)
    model = DetectionTransformer(
        backbone_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes=2,
        query_layers=4,
        num_queries=100,
    )
    print(model([image]))
