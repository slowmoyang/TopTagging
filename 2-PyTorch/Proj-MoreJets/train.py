from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from collections import OrderedDict
import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torch4hep.utils.misc import Directory
from torch4hep.projects.toptagging.utils import find_good_state

from dataset import get_data_loader
from model import Classifier
from model import init_weights

def get_data(batch, device):
    data = {
        "x_jet": [batch["x_jet{}".format(i)].to(device) for i in range(10)],
        "x_con_kin": [batch["x_con_kin{}".format(i)].to(device) for i in range(10)],
        "x_con_type": [batch["x_con_type{}".format(i)].to(device) for i in range(10)],
        "jet_mask": batch["jet_mask"].to(device),
        "con_mask": [batch["con_mask{}".format(i)].to(device) for i in range(10)]
    }
    return data


def train(model, data_loader, optimizer, criterion, device, logger):
    model.train()

    num_batch = len(data_loader)
    for batch_idx, batch in enumerate(data_loader, 0):
        data = get_data(batch, device)
        target = batch["y_true"].to(device)

        optimizer.zero_grad()

        y_score = model(**data)

        loss = criterion(y_score, target)
        loss.backward()
        optimizer.step()

        if (batch_idx % 128  == 0) or (batch_idx == (num_batch -1)):
            with torch.no_grad():
                y_pred = y_score > 0.5

            target = target.long().cpu().data.numpy()
            y_score = y_score.cpu().data.numpy()[:, 1]
            y_pred = y_pred.long().cpu().data.numpy()[:, 1]

            roc_auc = roc_auc_score(target, y_score)
            accuracy = accuracy_score(target, y_pred)

            logger.info("({:4d}/{}) Loss: {:.4f} | Acc.: {:.2f}% | AUC: {:.3f}".format(
                batch_idx, num_batch,
                loss.item(), 100 * accuracy, roc_auc))



def validate(model, data_loader, device, logger):
    logger.info("Start to validate model")

    loss_list = []
    y_true_list = []
    y_score_list = []
    y_pred_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = get_data(batch, device)
            target = batch["y_true"].to(device)

            y_score = model(**data)
            y_pred = y_score > 0.5

            # sum up batch loss
            loss = F.cross_entropy(input=y_score, target=target,
                                   reduction="elementwise_mean")

            loss = loss.item()
            target = target.cpu().data.numpy()
            y_score = y_score.cpu().data.numpy()[:, 1]
            y_pred = y_pred.long().cpu().data.numpy()[:, 1]

            loss_list.append(loss)
            y_true_list.append(target)
            y_score_list.append(y_score)
            y_pred_list.append(y_pred)

    y_true_list = np.concatenate(y_true_list)
    y_score_list = np.concatenate(y_score_list)
    y_pred_list = np.concatenate(y_pred_list)

    avg_loss = np.mean(loss_list)
    accuracy = accuracy_score(y_true=y_true_list,
                              y_pred=y_pred_list)
    auc = roc_auc_score(y_true=y_true_list,
                        y_score=y_score_list)

    logger.info('\tLoss: {:.4f} | Acc.: {:.2f}% | ROC AUC: {:3f}\n'.format(
        avg_loss, 100 * accuracy, auc))

    results = OrderedDict([
        ("loss", avg_loss),
        ("accuracy", accuracy),
        ("auc", auc)
    ])

    return results


def save_model(model, log_dir, epoch, results):
    # stringized results
    suffix = ""
    for key, value in results.iteritems():
        if isinstance(value, float):
            suffix += "_{}-{:.4f}".format(key, value)
        elif isinstance(value, int):
            suffix += "_{}-{:d}".format(key, value)
        else:
            suffix += "_{}-{}".format(key, value)
    path = "model_epoch-{:03d}{}.pth.tar".format(epoch, suffix)
    path = log_dir.state_dict.concat(path)
    torch.save(model.state_dict(), path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-set', type=str, dest="train_set",
                        default="/store/slowmoyang/TopTagging/toptagging-training.root")
    parser.add_argument('--valid-set', type=str, dest="valid_set",
                        default="/store/slowmoyang/TopTagging/toptagging-validation.root")
    parser.add_argument('--test-set', dest="test_set",
                        default="/store/slowmoyang/TopTagging/toptagging-test.root",
                        type=str)
    parser.add_argument('--batch-size', dest="batch_size",
                        default=128, type=int,
                        help='batch size')
    parser.add_argument('--test-batch-size', type=int, default=2048,
                        dest="test_batch_size",
                        help='batch size for test and validation')
    parser.add_argument('--epoch', dest="num_epochs", default=2, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--lr', default=0.005, type=float, 
                        help='learning rate, default=0.005')
    parser.add_argument("--logdir", dest="log_dir",
                        default="./logs/untitled", type=str,
                        help="the path to direactory")
    parser.add_argument("--verbose", default=True, type=bool)
    args = parser.parse_args()


    #####################################
    #
    ######################################
    log_dir = Directory(args.log_dir, create=True)
    log_dir.mkdir("state_dict")
    log_dir.mkdir("validation")
    log_dir.mkdir("roc_curve")

    ##################################################
    # Logger
    ################################################
    logger = logging.getLogger("TopTagging")
    logger.setLevel(logging.INFO)

    format_str = '[%(asctime)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, date_format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file_path = log_dir.concat("log.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    ###############################
    #
    ####################################
    device = torch.device("cuda:0")

    ######################################
    # Data Loader
    ########################################
    train_loader = get_data_loader(path=args.train_set,
                                   batch_size=args.batch_size)

    valid_loader = get_data_loader(path=args.valid_set,
                                   batch_size=args.test_batch_size)

    test_loader = get_data_loader(path=args.test_set,
                                  batch_size=args.test_batch_size)

    #####################
    # Model
    ######################
    model = Classifier()
    model.apply(init_weights)
    model.cuda(device)

    if args.verbose:
        logger.info(model)

    ##################################
    # Objective, optimizer,
    ##################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ################################
    # Callbacks
    ################################
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=args.verbose)

    ########################################
    # NOTE
    #######################################
    for epoch in xrange(1, args.num_epochs + 1):
        logger.info("Epoch: [{:d}/{:d}]".format(epoch, args.num_epochs))

        train(model=model,
              data_loader=train_loader,
              optimizer=optimizer,
              criterion=criterion,
              device=device,
              logger=logger)

        results = validate(model, valid_loader, device, logger)

        # Callbacks
        scheduler.step(results["loss"])

        save_model(model, log_dir, epoch, results)

    good_states = find_good_state(log_dir.state_dict.path)
    for each in good_states:
        model.load_state_dict(torch.load(each))
        # evaluate(model, test_loader, log_dir)
        
    logger.info("END")
    

if __name__ == "__main__":
    main()
