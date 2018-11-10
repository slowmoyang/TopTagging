from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

import warnings
from numbers import Real


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.GRU):
        nn.init.xavier_uniform_(module.weight_ih_l0)
        nn.init.xavier_uniform_(module.weight_hh_l0)
        nn.init.constant_(module.bias_ih_l0, 0.0)
        nn.init.constant_(module.bias_hh_l0, 0.0)
    else:
        pass


class Classifier(nn.Module):
    def __init__(self,
                 dim_jet=4,
                 dim_constituent=3):
        super(Classifier, self).__init__()

        jet_out_channels = 64
        con_out_channels = 128
        final_out_channels = 32
        embedding_dim = 3

        self._linear_jet = nn.Linear(dim_jet, jet_out_channels)
        # self._bn_jet = nn.BatchNorm1d(num_features=jet_out_channels)

        self._embedding = nn.Embedding(
            num_embeddings=9,
            embedding_dim=embedding_dim)

        self._gru_con = nn.GRU(
            dim_constituent + embedding_dim,
            con_out_channels,
            batch_first=True)

        self._gru = nn.GRU(
            jet_out_channels + con_out_channels,
            final_out_channels,
            batch_first=True)

        # self._dropout = nn.Dropout(0.5)
        self._linear = nn.Linear(final_out_channels, 2)


    def forward(self, x_jet, x_con_kin, x_con_type, jet_mask, con_mask):
        h_jet = [self._linear_jet(each) for each in x_jet]
        h_jet = [F.elu(each) for each in h_jet]
        # h_jet = [self._bn_jet(each) for each in h_jet]

        embed = [self._embedding(each) for each in x_con_type]
        x_con = [torch.cat(each, dim=-1) for each in zip(x_con_kin, embed)]
        h_con = [self._gru_con(each)[0] for each in x_con]

        con_mask = [each.view(-1, 1, 1) for each in con_mask]
        con_mask = [mask.expand(tensor.size(0), 1, tensor.size(2)) for mask, tensor in zip(con_mask, h_con)]

        h_con = [torch.gather(tensor, 1, mask) for mask, tensor in zip(con_mask, h_con)]
        h_con = [each.squeeze() for each in h_con]

        hidden = [torch.cat(each, dim=-1) for each in zip(h_jet, h_con)]
        hidden = torch.stack(hidden, dim=1)

        hidden, _ = self._gru(hidden)

        # 
        jet_mask = jet_mask.view(-1, 1, 1)
        jet_mask = jet_mask.expand(hidden.size(0), 1, hidden.size(2))

        hidden = torch.gather(hidden, 1, jet_mask)
        hidden = hidden.squeeze()

        # hidden = self._dropout(hidden) 
        logits = self._linear(hidden)
        logits = F.softmax(logits, dim=-1)
        return logits
        



def _test():
    from dataset import SixJetsDataset
    from torch.utils.data import DataLoader
    dset = SixJetsDataset("/store/slowmoyang/TopTagging/toptagging-test.root")
    data_iter = iter(DataLoader(dset, batch_size=128))
    batch = data_iter.next()

    model = Classifier()
    model.apply(init_weights)
    y_score = model(
        x_jet=[batch["x_jet{}".format(i)] for i in range(10)],
        x_con_kin=[batch["x_con_kin{}".format(i)] for i in range(10)],
        x_con_type=[batch["x_con_type{}".format(i)] for i in range(10)],
        jet_mask=batch["jet_mask"],
        con_mask=[batch["con_mask{}".format(i)] for i in range(10)])
    print("y_score: {}".format(y_score.shape))
    loss = F.cross_entropy(input=y_score, target=batch["y_true"])
    print(loss.mean())

    device = torch.device("cuda:0")
    model.cuda(device)
    y_score = model(
        x_jet=[batch["x_jet{}".format(i)].to(device) for i in range(10)],
        x_con_kin=[batch["x_con_kin{}".format(i)].to(device) for i in range(10)],
        x_con_type=[batch["x_con_type{}".format(i)].to(device) for i in range(10)],
        jet_mask=batch["jet_mask"].to(device),
        con_mask=[batch["con_mask{}".format(i)].to(device) for i in range(10)])
   
if __name__ == "__main__":
    _test()
