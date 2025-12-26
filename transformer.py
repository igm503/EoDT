import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm


class DetectionTransformer(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        num_classes: int = 80,
        num_queries: int = 300,
        query_layers: int = 4,
    ):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.preprocess = AutoImageProcessor.from_pretrained(backbone_name)
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        hidden_dim = self.backbone.config.hidden_size
        
        self.bbox_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        
        self.cls_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        
        # Find encoder layers
        if hasattr(self.backbone, "encoder"):
            self.encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layer"):
            self.encoder_layers = self.backbone.layer
        else:
            raise ValueError("Cannot find encoder layers in backbone")
        
        self.first_query_layer = max(0, len(self.encoder_layers) - query_layers)
        
        # Initialize bbox head for small outputs
        nn.init.constant_(self.bbox_mlp[-1].weight, 0)
        nn.init.constant_(self.bbox_mlp[-1].bias, 0)
        
        # Initialize cls head with prior for focal loss
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_mlp[-1].bias, bias_value)
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        pos = self.backbone.rope_embeddings(x)
        x = self.backbone.embeddings(x)

        for i, layer in enumerate(self.backbone.layer):
            if i == self.first_query_layer:
                query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                x = torch.cat([query_tokens, x], dim=1)
            x = layer(x, position_embeddings=pos)

        # Final norm
        if hasattr(self.backbone, "layernorm"):
            x = self.backbone.layernorm(x)
        elif hasattr(self.backbone, "norm"):
            x = self.backbone.norm(x)
        elif hasattr(self.backbone.encoder, "layer_norm"):
            x = self.backbone.encoder.layer_norm(x)

        final_queries = x[:, :self.num_queries, :]

        bboxes = nn.functional.sigmoid(self.bbox_mlp(final_queries))
        scores = self.cls_mlp(final_queries)
        
        return {"pred_boxes": bboxes, "pred_logits": scores}


if __name__ == "__main__":
    image = torch.randn(1, 3, 640, 640)
    model = DetectionTransformer(
        backbone_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes=2,
        query_layers=4,
        num_queries=100,
    )
    model.to("cuda")
    image.to("cuda")
    for i in tqdm(range(100)):
        model([image] * 16)
    # print(model([image]))
