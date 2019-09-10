# coding=utf-8
from pt_pack.modules.layers.base_layers import Layer
import torch.nn as nn
import torch
import numpy as np





class CgsGraphLearnerLayer(Layer):
    cls_name = 'cgs_graph_learner_layer'

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, hid_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        B, K, C = graph_nodes.shape
        graph_nodes = graph_nodes.view(-1, C)
        h = self.proj_l(graph_nodes)
        # outer product
        h = h.view(B, K, -1)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        return adjacency_matrix


class CgsGraphConvLayer(Layer):
    '''
    Implementation of: https://arxiv.org/pdf/1611.08402.pdf where we consider
    a fixed sized neighbourhood of nodes for each feature
    '''
    cls_name = 'cgs_graph_conv_layer'

    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_num,
                 bias=False):
        super().__init__()
        self.kernel_num = kernel_num
        self.in_feat_dim = in_dim
        self.out_feat_dim = out_dim
        self.bias = bias

        # Convolution filters weights
        self.linear_layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim // kernel_num, bias=bias) for _ in range(kernel_num)]
        )
        self.relu_l = nn.ReLU(inplace=True)

        # Parameters of the Gaussian kernels
        self.mean_rho = nn.Parameter(torch.Tensor(kernel_num, 1))
        self.mean_theta = nn.Parameter(torch.Tensor(kernel_num, 1))
        self.precision_rho = nn.Parameter(torch.Tensor(kernel_num, 1))
        self.precision_theta = nn.Parameter(torch.Tensor(kernel_num, 1))
        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

    def forward(self, nh_feat, nh_coord):
        '''
        ## Inputs:
        - nh_feat (batch_size, K, neighbourhood_size, in_feat_dim)
        - nh_coord (batch_size, K, neighbourhood_size, coordinate_dim)
        ## Returns:
        - convolved_features (batch_size, K, neighbourhood_size, out_feat_dim)
        '''
        B, K, nh_size, C = nh_feat.shape

        # compute pseudo coordinate kernel weights
        weights = self.get_gaussian_weights(nh_coord) # b*k*nh_size, k_num
        weights = weights.view(B*K, nh_size, self.kernel_num)  # b*k, nh_size, k_num

        # compute convolved features
        nh_feat = nh_feat.view(B*K, nh_size, -1)
        convolved_features = self.convolution(nh_feat, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)  # b, k, c
        conv_feats = self.relu_l(convolved_features)
        return conv_feats

    def get_gaussian_weights(self, pseudo_coord):
        '''
        ## Inputs:
        - pseudo_coord (batch_size, K, neighbourhood_size, pseudo_coord_dim)
        ## Returns:
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        '''

        # compute rho weights
        diff = (pseudo_coord[:, :, :, 0].contiguous().view(-1, 1) - self.mean_rho.view(1, -1))**2  # b*K*nh_size, kernel_num
        weights_rho = torch.exp(-0.5 * diff / (1e-14 + self.precision_rho.view(1, -1)**2))

        # compute theta weights
        first_angle = torch.abs(pseudo_coord[:, :, :, 1].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle)**2) / (1e-14 + self.precision_theta.view(1, -1)**2))
        weights = weights_rho * weights_theta

        # print(self.mean_rho)

        if torch.isnan(weights).sum():
            print('tttttt')
        weights[(weights != weights).data] = 0

        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        return weights

    def convolution(self, nh_feat, nh_weight):
        '''
        ## Inputs:
        - neighbourhood (batch_size*K, neighbourhood_size, in_feat_dim)
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        ## Returns:
        - convolved_features (batch_size*K, out_feat_dim)
        '''
        # patch operator
        weighted_nh_feat = torch.bmm(nh_weight.transpose(1, 2), nh_feat) # b*k, kernel_num, C

        # convolutions
        weighted_nh_feat = [self.linear_layers[i](weighted_nh_feat[:, i]) for i in range(self.kernel_num)]
        convolved_features = torch.cat([i.unsqueeze(1) for i in weighted_nh_feat], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features
