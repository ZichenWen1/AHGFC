import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.init as init
from utils import remove_self_loop


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, step=2., droprate=0.3):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.droprate = droprate
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        tmp_dim = hidden_dim
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(tmp_dim, output_dim))
            else:
                tmp_hidden_dim = tmp_dim
                tmp_dim = int(tmp_dim / step)
                if tmp_dim < output_dim:
                    tmp_dim = tmp_hidden_dim
                self.enc.append(nn.Linear(tmp_hidden_dim, tmp_dim))
        # self.init_weights()

    def forward(self, x):
        z = self.encode(x)
        return z

    def encode(self, x):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if self.droprate:
                    h = torch.dropout(h, self.droprate, train=self.training)
                h = layer(h)
            else:
                if self.droprate:
                    h = torch.dropout(h, self.droprate, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

# LatentMappingLayer
class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.enc.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h

# EnDecoder
class EnDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(EnDecoder, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = torch.sigmoid(self.dec_f(z_norm, dropout))
        return x_pred, z_norm

# EnDecoder for pretraining
class EnDecoder_pretrain(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(EnDecoder_pretrain, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = torch.sigmoid(self.dec_f(z_norm, dropout))
        a_pred = torch.sigmoid(torch.mm(z, z.t()))
        return a_pred, x_pred, z_norm

class GraphEnc(nn.Module):
    def __init__(self, order=2):
        super(GraphEnc, self).__init__()
        self.order = order

    def forward(self, x, adj, rs, order=None):
        if order is not None:
            self.order = order
        adj = self.normalize_adj(adj)
        z = self.message_passing_global(x, adj, rs)
        return z

    def message_passing_global(self, x, adj, rs):
        h = x
        I = torch.eye(adj.shape[0]).to(adj.device)
        temp = I - adj
        for i in range(self.order):
            h = rs * torch.matmul(adj, h) + (1 - rs) * torch.matmul(temp, h)
            # + (1 * x) # TODO：temporary change！
            # h = torch.matmul(adj, h)  # TODO: ablation exp
            # h = 1.0 * torch.matmul(adj, h) + 1.0 * torch.matmul(temp, h)   # TODO: for rebuttal
        return h

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x

    def normalize_adj_symmetry(self, x):
        D = x.sum(1).detach().clone()
        D_sqrt_inv = D.pow(-0.5).flatten()
        D_sqrt_inv = D_sqrt_inv.reshape((x.shape[0], -1))
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.
        x = D_sqrt_inv * x
        x = x * D_sqrt_inv.T
        return x


class MixedFilter(nn.Module) :
    def __init__(self, order):
        super(MixedFilter, self).__init__()
        self.order = order

    def forward(self, X, S, S_bar, rs, order=None):
        if order is not None:
            self.order = order
        H_low = self.LowPassFilter(X, S)
        H_high = self.HighPassFilter(X, S_bar)
        H = rs * H_low + (1 - rs) * H_high
        return H


    def LowPassFilter(self, X, S, p=0.5):
        I = torch.eye(S.shape[0]).to(S.device)
        # S = S + I
        S = self.normalize_matrix(S)
        L_S = I - S
        H_low = X.clone()
        for i in range(self.order):
            H_low = (I - p * L_S).matmul(H_low)
        return H_low


    def HighPassFilter(self, X, S_bar, p=0.5):
        I = torch.eye(S_bar.shape[0]).to(S_bar.device)
        S_bar_ = remove_self_loop(S_bar)
        S_bar_ = self.normalize_matrix(S_bar_)
        L_S_bar = I - S_bar_

        H_high = X.clone()
        for i in range(self.order):
            H_high = p * L_S_bar.matmul(H_high)

        return H_high

    def normalize_matrix(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1e-100).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x


class core_model(nn.Module):
    def __init__(self, input_dim_x, hidden_dim_x, output_dim_x, input_dim_a, hidden_dim_a, output_dim_a,
                 class_num, num_layers_x=2, step_x=2, num_layers_a=2, step_a=2, k=10):
        super(core_model, self).__init__()
        self.k = k

        self.endecs_x = EnDecoder(input_dim_x, hidden_dim_x, output_dim_x)
        self.endecs_a = EnDecoder(input_dim_a, hidden_dim_a, output_dim_a)

        self.cluster_layer_x = Parameter(torch.Tensor(class_num, output_dim_x))
        self.register_parameter('centroid_x', self.cluster_layer_x)
        self.cluster_layer_a = Parameter(torch.Tensor(class_num, output_dim_a))
        self.register_parameter('centroid_a', self.cluster_layer_a)

    def forward(self, x, adj):
        x_pred, z_from_x = self.endecs_x(x)
        # z_from_x_norm = F.normalize(z_from_x, p=2, dim=-1) #??
        # print('z_x_type : {}'.format(z_from_x.dtype()))

        a_pred, z_from_a = self.endecs_a(adj)
        # print('z_a_type : {}'.format(z_from_a.dtype()))
        # z_from_a_norm = F.normalize(z_from_a, p=2, dim=-1) #??
        return  x_pred, a_pred, z_from_x, z_from_a

    def predict_distribution(self, z, layer, alpha=1.0):
        c = layer
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


class Model(nn.Module):
    def __init__(self,
                 input_dim_x, hidden_dim_x, output_dim_x,
                 input_dim_a, hidden_dim_a, output_dim_a,
                 input_dim_g, hidden_dim_g, output_dim_g,
                 class_num, node_num, view_num, order=2, num_layers_x=2, step_x=2, num_layers_a=4, step_a=2, k=10):
        super(Model, self).__init__()
        self.class_num = class_num
        self.node_num = node_num
        self.view_num = view_num
        self.order = order
        self.k = k

        self.core = nn.ModuleList([
            core_model(input_dim_x, hidden_dim_x, output_dim_x, input_dim_a, hidden_dim_a, output_dim_a,
                          class_num, num_layers_x, step_x, num_layers_a, step_a, k=k) for _ in range(view_num)
        ])

        self.graphencs = nn.ModuleList([
            GraphEnc(order=order) for _ in range(view_num)
        ])

        self.MixedFilter = nn.ModuleList([
            MixedFilter(order=order) for _ in range(view_num)
        ])

        self.cluster_layer = [Parameter(torch.Tensor(class_num, output_dim_g)) for _ in range(view_num)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, output_dim_g)))
        for i in range(view_num):
            self.register_parameter('centroid_{}'.format(i), self.cluster_layer[i])
        self.register_parameter('centroid_{}'.format(view_num), self.cluster_layer[view_num])

    def forward(self, xs, adjs, weights_h, rs, order=None):
            if order is not None:
                self.order = order

            Zs = []
            x_homos = []
            zx_norms = []
            za_norms = []
            x_preds = []
            a_preds = []
            hs = []
            Ss = []
            qgs = []

            for v in range(self.view_num):

                x_pred, a_pred, z_from_x, z_from_a = self.core[v](xs[v], adjs[v])
                zx_norms.append(z_from_x)
                za_norms.append(z_from_a)


                x_preds.append(x_pred)
                a_preds.append(a_pred)

                temp = torch.matmul(z_from_a, z_from_x.T)
                S = torch.mm(temp, temp.T)

                # discretization
                row_means = torch.mean(S, dim=1, keepdim=True)
                S_dis = torch.where(S > row_means, torch.tensor(1,device=S.device), torch.tensor(0,device=S.device))

                S = S_dis + torch.eye(S.shape[0], device=S.device) # add I
                S_bar = S_dis  # delete I


                # S = torch.mm(S, S.T)
                adjs_input = adjs[v] + torch.eye(adjs[v].shape[0], device=adjs[v].device)   # TODO:ablation exp

                # AX
                # A_X = temp + torch.eye(temp.shape[0], device=temp.device)   # TODO:ablation exp

                # graphencs
                h = self.graphencs[v](z_from_x, S, rs[v])  # TODO:ablation exp
                h = F.normalize(h, p=2, dim=-1)
                hs.append(h)

                qg = self.predict_distribution(h, v)
                qgs.append(qg)
                Ss.append(S)

            h_all = sum(weights_h[v] * hs[v] for v in range(self.view_num)) / sum(weights_h)
            qg = self.predict_distribution(h_all, -1)
            qgs.append(qg)

            return  x_preds, a_preds, qgs, hs, h_all, Ss, zx_norms, za_norms, x_homos

    def predict_distribution(self, z, v, alpha=1.0):
            c = self.cluster_layer[v]
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
            q = q.pow((alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            return q

    @staticmethod
    def target_distribution(q):
            weight = q ** 2 / q.sum(0)
            return (weight.t() / weight.sum(1)).t()




