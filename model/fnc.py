import torch
import torch.nn as nn


class FNC8s(nn.Module):
    def __init__(self, num_classes) -> None:
        super(FNC8s, self).__init__()

        