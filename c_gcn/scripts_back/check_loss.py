import torch.nn as nn

import torch

bce_loss = nn.BCEWithLogitsLoss()
margin_loss = nn.MultiLabelSoftMarginLoss()


label = torch.Tensor([0, 0, 0.3, 0, 0.7, 0.7]).view(1, 6)

x = torch.Tensor([0.2, 0.3, 0.5, 0.1, 0.8, 0.9]).view(1, 6)



print(bce_loss(x, label))
print(margin_loss(x, label))