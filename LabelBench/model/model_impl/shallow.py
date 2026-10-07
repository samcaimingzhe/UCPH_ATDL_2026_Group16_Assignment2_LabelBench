"""Single-hidden-layer proxy used in the official Figure 5 experiments."""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP

from LabelBench.skeleton.model_skeleton import register_model


class ModifiedShallow(nn.Module):
    def __init__(self, num_input, num_output, num_hidden=2, ret_emb=False):
        super().__init__()
        if num_hidden < 2:
            raise ValueError("num_hidden counts hidden plus output layers; must be >= 2")
        self.shallow_model = MLP(in_channels=num_input,
                                 hidden_channels=[num_input] * (num_hidden - 1),
                                 norm_layer=None, dropout=0.0)
        self.classifier = nn.Linear(num_input, num_output)
        self.ret_emb = ret_emb

    def forward(self, features, ret_features=False, **kwargs):
        features = F.relu(self.shallow_model(features))
        logits = self.classifier(features)
        if ret_features:
            return logits, features.detach()
        if self.ret_emb:
            return logits, features
        return logits


@register_model("shallow")
def init_shallow(config):
    return ModifiedShallow(config["input_dim"], config["num_output"],
                           config.get("num_hidden", 2), config.get("ret_emb", False))
