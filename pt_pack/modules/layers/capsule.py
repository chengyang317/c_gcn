# coding=utf-8
import torch
import torch.nn as nn


__all__ = ['CapsuleLayer']



class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    @classmethod
    def squash(cls, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        """

        :param x: b, route_num, c
        :return:
        """
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]  # cap_num, b, route_num, 1, c
            logits = torch.zeros(*priors.size()).type_as(x)
            for i in range(self.num_iterations):
                probs = logits.softmax(dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))  # cap_num, b, 1, 1, c
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [self.capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs.squeeze().transpose(0, 1).contiguous()


class CondCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations=3):
        super().__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x, cond):
        """

        :param x: b, route_num, c
        :param cond: b, c
        :return:
        """
        priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]  # cap_num, b, route_num, 1, c
        logits = torch.zeros(*priors.size()).type_as(x)
        for idx in range(self.num_iterations):
            if idx == 0:
                outputs = self.squash(priors).sum(dim=2, keepdim=True)
            else:
                pass
            probs = logits.softmax(dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))  # cap_num, b, 1, 1, c
            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

        return outputs.squeeze().transpose(0, 1).contiguous()

    def compute_prob(self, ):
        pass


if __name__ == '__main__':
    layer = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
    x = torch.ones(5, 256, 20, 20)
    y = layer(x)
    layer = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
    z = layer(y)