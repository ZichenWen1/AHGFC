import argparse
import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.backends import cudnn
from utils import load_data, cal_homo_ratio, rw_normalize, minmaxnormalize, normalize_weight
from models import Model, EnDecoder_pretrain
from evaluation import eva
import pandas as pd
from settings import get_settings
from Visualization import plt_tsne

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='texas', help='datasets: texas, chameleon, minesweeper, acm, acm00, acm01, acm02, acm03, acm04, acm05')
parser.add_argument('--train', type=bool, default=False, help='training mode')
parser.add_argument('--cuda_device', type=int, default=0, help='')
parser.add_argument('--use_cuda', type=bool, default=True, help='')
parser.add_argument('--pretrain', type=int, default=500, help='pretrain epochs')
parser.add_argument('--endecoder_lr', type=float, default=0.001, help='learning rate for autoencoder')
parser.add_argument('--endecoder_weight_decay', type=float, default=0.1, help='weight decay for autoencoder')
parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
args = parser.parse_args()


n = 0
train = args.train
if args.train:
    print('\n\033[1;32mTraining mode\033[0m')
else:
    print('\n\033[1;32mTest mode\033[0m')

dataset = args.dataset
use_cuda = args.use_cuda
cuda_device = args.cuda_device

# pretrain
pretrain = args.pretrain
endecoder_lr = args.endecoder_lr
endecoder_weight_decay = args.endecoder_weight_decay
endecoder_hidden_dim = args.endecoder_hidden_dim

# ------------------------------specific datasets settings---------------------------------------
settings = get_settings(dataset)
weight_soft_h = settings.weight_soft_h
path = settings.path
order = settings.order
# k_for_disc = settings.K

hidden_dim_x = settings.hidden_dim_x
output_dim_x = settings.output_dim_x
hidden_dim_a = settings.hidden_dim_a
output_dim_a = settings.output_dim_a
hidden_dim_g = settings.hidden_dim_g
output_dim_g = settings.output_dim_g
num_layers_x = settings.num_layers_x
num_layers_a = settings.num_layers_a

epoch = settings.epoch
patience = settings.patience
lr = settings.lr
weight_decay = settings.weight_decay

update_interval = settings.update_interval
random_seed = settings.random_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True


# loss weight
Weight_loss_re_x = settings.Weight_loss_re_x
Weight_loss_re_x_mse = settings.Weight_loss_re_x_mse
Weight_loss_re_a = settings.Weight_loss_re_a
Weight_loss_kl_g = settings.Weight_loss_kl_g

# loss weight print
print('\n\033[1;34mloss weight:\033[0m')
print('Weight_loss_re_x:{}'.format(Weight_loss_re_x))
print('Weight_loss_re_x_mse:{}'.format(Weight_loss_re_x_mse))
print('Weight_loss_re_a:{}'.format(Weight_loss_re_a))
print('Weight_loss_kl_g:{}'.format(Weight_loss_kl_g))
print(f"loss = {Weight_loss_re_x} * loss_re_x + {Weight_loss_re_x_mse} * loss_re_x_mse + {Weight_loss_re_a} * loss_re_a + {Weight_loss_kl_g} * loss_kl_g")


# parameter print
print('\n\033[1;34mparameter settings:\033[0m')
print('learning rate:{}'.format(lr))
print('weight decay:{}'.format(weight_decay))
print('hidden_dim_x:{}'.format(hidden_dim_x))
print('output_dim_x:{}'.format(output_dim_x))
print('hidden_dim_a:{}'.format(hidden_dim_a))
print('output_dim_a:{}'.format(output_dim_a))
print('hidden_dim_g:{}'.format(hidden_dim_g))
print('output_dim_g:{}'.format(output_dim_g))
print()


# load data
labels, adjs, adjs_labels, shared_feature, shared_feature_label, num_graph = load_data(dataset, path)

# dataset information
rs = []
for v in range(num_graph):
    r, homo = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
    rs.append(r)     # true homo ratio form true labels
    print(r, homo)   # calculate homo ratio

# print dataset information
print('\n\033[1;34mdataset information:\033[0m')
print('dataset:{}'.format(dataset))
print('nodes: {}'.format(shared_feature_label.shape[0]))
print('features: {}'.format(shared_feature_label.shape[1]))
print('class: {}'.format(labels.max() + 1))

feat_dim = shared_feature.shape[1]

class_num = labels.max().item() + 1
y = labels.cpu().numpy()
node_num = shared_feature.shape[0]

xs = []
As = []
for v in range(num_graph):
    xs.append(shared_feature_label)
    As.append(adjs_labels[v])

# endecoder initial
endecoder = EnDecoder_pretrain(feat_dim, endecoder_hidden_dim, class_num)
# model initial
model = Model(feat_dim, hidden_dim_x, output_dim_x,
              node_num, hidden_dim_a, output_dim_a,
              feat_dim, hidden_dim_g, output_dim_g,
              class_num, node_num, num_graph, order=order, k=None,
              num_layers_x=num_layers_x, num_layers_a=num_layers_a)

if use_cuda:
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(cuda_device)
    torch.cuda.manual_seed(random_seed)
    endecoder = endecoder.cuda()
    model = model.cuda()
    adjs_labels = [a.cuda() for a in adjs_labels]
    adjs = [a.cuda() for a in adjs]
    shared_feature = shared_feature.cuda()
    shared_feature_label = shared_feature_label.cuda()
    xs = [x.cuda() for x in xs]
    As = [a.cuda() for a in As]
device = adjs_labels[0].device




if train:
    # =============================================== pretrain endecoder ============================
    # print('shared_feature_label for clustering...')
    # print('Begin pretrain...')
    print('\033[1;34mBegin pretrain...\033[0m')
    kmeans = KMeans(n_clusters=class_num, n_init=3, random_state=42)
    y_pred = kmeans.fit_predict(shared_feature_label.data.cpu().numpy())
    eva(y, y_pred, 'Kz')
    print()

    optimizer_endecoder = torch.optim.Adam(endecoder.parameters(), lr=endecoder_lr, weight_decay=endecoder_weight_decay)

    for epoch_num in range(pretrain):
        endecoder.train()
        loss_re = 0.
        loss_a = 0.

        a_pred, x_pred, z_norm = endecoder(shared_feature)
        for v in range(num_graph):
            loss_a += F.binary_cross_entropy(a_pred, adjs_labels[v])
        loss_re += F.binary_cross_entropy(x_pred, shared_feature_label)

        loss = loss_re + loss_a
        optimizer_endecoder.zero_grad()
        loss.backward()
        optimizer_endecoder.step()
        print('epoch: {}, loss:{}, loss_re:{}, loss_a: {}'.format(epoch_num, loss, loss_re, loss_a))

        if epoch_num == pretrain - 1:
            print('Pretrain complete...')
            kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=42)
            y_pred = kmeans.fit_predict(z_norm.data.cpu().numpy())
            eva(y, y_pred, 'Kz')
            break

    homo_rate = []  # 初始化同配率
    for v in range(num_graph):
        r, homo = cal_homo_ratio(adjs_labels[v].cpu().numpy(), np.asarray(y_pred), self_loop=True)
        homo_rate.append(r)

    # =========================================Train============================================
    print('\033[1;34mBegin train...\033[0m')
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bad_count = 0
    best_loss = 100
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0
    loss = 0.
    kl_step = 1.
    kl_max = 10000
    l = 0.0

    XXs = []
    AAs = []


    for v in range(num_graph):
        x = minmaxnormalize(rw_normalize(torch.mm(shared_feature_label, shared_feature_label.T)))
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        XXs.append(x)

        a = minmaxnormalize(rw_normalize(torch.mm(adjs_labels[v], adjs_labels[v].T)))
        AAs.append(a)

        weighh = [1e-12 for i in range(num_graph)]
        weights_h = normalize_weight(weighh)
        weights_S = []
    for v in range(num_graph):
        weightss = [1e-12, 1e-12]
        weightsS = normalize_weight(weightss)
        weights_S.append(weightsS)

    with torch.no_grad():
        model.eval()
        x_preds, a_preds, qgs, hs, h_all, Ss, zx_norms, za_norms, x_homos = model(xs, adjs_labels, weights_h, homo_rate)
        kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=42)

        # homo_rates
        # homo_rates = []

        for v in range(num_graph):
            y_pred_g = kmeans.fit_predict(hs[v].data.cpu().numpy())
            model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
        y_pred_g = kmeans.fit_predict(hs[-1].data.cpu().numpy())
        model.cluster_layer[-1].data = torch.tensor(kmeans.cluster_centers_).to(device)

    losses = []
    for epoch_num in range(epoch):
        model.train()
        loss = 0.
        loss_re_x = 0.
        loss_re_a = 0.
        loss_kl_g = 0.
        loss_re_x_mse = 0.

        x_preds, a_preds, qgs, hs, h_all, Ss, zx_norms, za_norms, x_homos = model(xs, adjs_labels, weights_h, homo_rate)
        kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=42)
        y_prim = kmeans.fit_predict(h_all.detach().cpu().numpy())
        pseudo_label = y_prim

        for v in range(num_graph):
            r, homo = cal_homo_ratio(adjs_labels[v].cpu().numpy(), np.asarray(pseudo_label), self_loop=True)
            homo_rate[v] = r

        for v in range(num_graph):
            y_pred = kmeans.fit_predict(hs[v].detach().cpu().numpy())
            a = eva(y_prim, y_pred, visible=False, metrics='acc')
            weighh[v] = a
        weights_h = normalize_weight(weighh, p=weight_soft_h)

        pgh = model.target_distribution(qgs[-1])

        loss_kl_g += F.kl_div(qgs[-1].log(), pgh, reduction='batchmean')

        for v in range(num_graph):
            loss_re_x += F.binary_cross_entropy(x_preds[v], xs[v])
            # add mse
            loss_re_x_mse += F.mse_loss(x_preds[v], xs[v])
            loss_re_a += F.binary_cross_entropy(a_preds[v], As[v])
            pg = model.target_distribution(qgs[v])
            loss_kl_g += F.kl_div(qgs[v].log(), pg, reduction='batchmean')
            loss_kl_g += F.kl_div(qgs[v].log(), pgh, reduction='batchmean')
        if l < kl_max:
            l = kl_step * epoch_num
        else:
            l = kl_max
        loss_kl_g *= l

        loss += Weight_loss_re_x * loss_re_x + Weight_loss_re_x_mse * loss_re_x_mse + Weight_loss_re_a * loss_re_a + Weight_loss_kl_g * loss_kl_g


        losses.append(loss.item())

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # print loss
        # print(
        #     'epoch: {}, loss: {:.4f}, loss_re_x: {:.4f}, loss_re_x_mse: {:.4f}, loss_re_a:{:.4f}, loss_kl_g: {:.4f}, badcount: {}'.format(
        #         epoch_num, loss, loss_re_x, loss_re_a, loss_re_x_mse, loss_kl_g, bad_count
        #     ))

        if epoch_num % update_interval == 0:
            model.eval()
            x_preds, a_preds, qgs, hs, h_all, Ss, zx_norms, za_norms, x_homos = model(xs, adjs_labels, weights_h, homo_rate)
            kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=42)
            y_eval = kmeans.fit_predict(h_all.detach().cpu().numpy())
            nmi, acc, ari, f1 = eva(y, y_eval, str(epoch_num) + 'Kz', visible=False)

        if acc > best_acc:
            if os.path.exists('./pkl/AHGFC_{}_weight_decay_{}_lr{}_acc{:.4f}.pkl'.format(dataset, weight_decay, lr, best_acc)):
                os.remove('./pkl/AHGFC_{}_weight_decay_{}_lr{}_acc{:.4f}.pkl'.format(dataset, weight_decay, lr,  best_acc))
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            best_loss = loss
            bad_count = 0

            torch.save({'state_dict': model.state_dict(),
                        'state_dict_endecoder': endecoder.state_dict(),
                        'xs': xs,
                        'adjs_labels': adjs_labels,
                        'h_all': h_all,
                        'weights_S': weights_S,
                        'weights_h': weights_h,
                        'datasets': dataset,
                        'lrs': lr,
                        'weight_decays': weight_decay,
                        # 'K': k_for_disc,
                        'homo_rate': homo_rate},
                        './pkl/AHGFC_{}_weight_decay_{}_lr{}_acc{:.4f}.pkl'.format(dataset, weight_decay, lr,  best_acc))

            print(
                'best acc:{:.4f}, best nmi:{:.4f}, best ari:{:.4f}, best f1:{:.4f}, best loss:{:.4f}, bestepoch:{}'.format(
                    best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
        else:
            bad_count += 1

        if bad_count >= patience:
            print(
                'complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
                    best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
            print()
            break

    # record the best model
    columns = ['dataset', 'acc', 'nmi', 'ari', 'f1', 'epoch', 'lr', 'weight_decay', 'hidden_dim_x', 'hidden_dim_a', 'hidden_dim_g', 'order', 'Weight_loss_re_x', 'Weight_loss_re_x_mse', 'Weight_loss_re_a', 'Weight_loss_kl_g']
    dt = np.asarray(
        [dataset, best_acc, best_nmi, best_ari, best_f1, best_epoch, lr, weight_decay, hidden_dim_x, hidden_dim_a, hidden_dim_g, order, Weight_loss_re_x, Weight_loss_re_x_mse, Weight_loss_re_a, Weight_loss_kl_g]
    ).reshape(1, -1)
    df = pd.DataFrame(dt, columns=columns)
    if n == 0:
        head = True
    else:
        head = False
    df.to_csv('./result.csv', index=False, header=head, mode='a')
    n += 1

if not train:
    model_name = 'AHGFC_{}'.format(dataset)
else:
    model_name = 'AHGFC_{}_weight_decay_{}_lr{}_acc{:.4f}'.format(dataset, weight_decay, lr,  best_acc)

best_model = torch.load('./pkl/'+model_name+'.pkl', map_location=shared_feature.device)

state_dic = best_model['state_dict']



weights_S = best_model['weights_S']
weights_h = best_model['weights_h']
homo_rate = best_model['homo_rate']

model.load_state_dict(state_dic)

model.eval()
with torch.no_grad():
    x_preds, a_preds, qgs, hs, h_all, Ss, zx_norms, za_norms, x_homos = model(xs, adjs_labels, weights_h, homo_rate)
    kmeans = KMeans(n_clusters=class_num, n_init=15, random_state=42)
    y_eval = kmeans.fit_predict(h_all.detach().cpu().numpy())
    nmi, acc, ari, f1 = eva(y, y_eval, 'Final Kz')

print('\033[1;34mTest complete...\033[0m')




