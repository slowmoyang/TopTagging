from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import ROOT

from torch.utils.data import DataLoader

from torch4hep.datasets import BaseTreeDataset
from torch4hep.preprocessing.sequence import SeqLenAdjuster


class SixJetsDataset(BaseTreeDataset):
    def __init__(self, path, max_lengths=[50]*10, tree_name="test", transform=None):
        super(SixJetsDataset, self).__init__(path, tree_name, transform)
        self.seqlen_adjusters = [SeqLenAdjuster(max_len=max_len) for max_len in max_lengths]
        self._max_lengths = max_lengths

    def __getitem__(self, idx):
        self._tree.GetEntry(idx)
        example = {}

        jet_mask = len(self._tree.jet_b_tag)
        for jet_idx in range(10):
            if jet_idx < jet_mask:
                four_momenta = self._tree.jet_p4[jet_idx]
                btag = self._tree.jet_b_tag[jet_idx]
                jet = [four_momenta.Pt(), four_momenta.Eta(), four_momenta.Phi(), btag]
                jet = np.array(jet, dtype=np.float32)
                example["x_jet{}".format(jet_idx)] = jet

                # the kinematic variables
                pt = np.array(self._tree.constituent_pt[jet_idx], dtype=np.float32)
                # relative pt
                # pt = pt / jet[0]
                deta = np.array(self._tree.constituent_deta[jet_idx], dtype=np.float32)
                dphi = np.array(self._tree.constituent_dphi[jet_idx], dtype=np.float32)
                con_kin = np.stack([pt, deta, dphi]).T

                # particle type id
                # type id starts from -3, but embedding requries non-negative numbers
                con_type = np.array(self._tree.constituent_type[jet_idx], dtype=np.int64) + 4

                # Sort constituents in the pT-order
                pt_order = np.argsort(pt)[::-1]
                con_kin = con_kin[pt_order]
                con_type = con_type[pt_order]

                #
                con_kin = self.seqlen_adjusters[jet_idx](con_kin)
                con_type = self.seqlen_adjusters[jet_idx](con_type)
                example["x_con_kin{}".format(jet_idx)] = con_kin
                example["x_con_type{}".format(jet_idx)] = con_type

                # FIXME it is not mask
                num_cons = len(pt)
                if num_cons <= self._max_lengths[jet_idx]:
                    con_mask = num_cons - 1
                else:
                    con_mask = 0
                example["con_mask{}".format(jet_idx)] = con_mask

            else:
                example["con_mask{}".format(jet_idx)] = np.int64(0)
                example["x_con_kin{}".format(jet_idx)] = np.zeros(shape=(50, 3), dtype=np.float32)
                example["x_con_type{}".format(jet_idx)] = np.zeros(shape=(50, ), dtype=np.int64)
                example["x_jet{}".format(jet_idx)] = np.zeros(shape=(4,), dtype=np.float32)

        example["y_true"] = np.int64(self._tree.label)
        example["jet_mask"] = np.int64(jet_mask) - 1 if jet_mask <= 10 else np.int64(1)

        if self._transform is not None:
            example = self._transform(example)

        return example

def get_data_loader(path, batch_size):
    dataset = SixJetsDataset(path=path)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return data_loader


def _test():
    import os
    host_name = os.environ["HOSTNAME"]
    if host_name == "cms05.sscc.uos.ac.kr":
        path = "/store/slowmoyang/TopTagging/toptagging-test.root"
    elif host_name == "gate2.sscc.uos.ac.kr":
        path = "/home/scratch/slowmoyang/TopTagging/toptagging-test.root"
    elif host_name == "cms-gpu01.sdfarm.kr":
        raise NotImplementedError
    else:
        raise EnvironmentError

    dset = SixJetsDataset(path)
    data_loader = DataLoader(dset, batch_size=20)
    data_iter = iter(data_loader)
    batch = data_iter.next()
    for key, tensor in batch.iteritems():
        print(key, tensor.shape, tensor.dtype)
    print(batch["y_true"])
    return data_iter

if __name__ == "__main__":
    _test()
