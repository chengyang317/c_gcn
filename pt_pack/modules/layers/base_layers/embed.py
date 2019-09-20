import torch.nn as nn
import torch
import torch.nn.functional as F


__all__ = ['Embedding']


class Embedding(nn.Module):
    def __init__(self, emb_init, add_pad=True):
        super().__init__()
        self.add_pad = add_pad
        self.embed_para = nn.Parameter(torch.Tensor(emb_init), requires_grad=True)

    def forward(self, q_idx):
        if self.add_pad:
            embedding_para = torch.cat(
                (torch.zeros(1, self.embed_para.size(-1), device=self.embed_para.device), self.embed_para)
            )
        else:
            embedding_para = self.embed_para
        return F.embedding(q_idx, embedding_para)




