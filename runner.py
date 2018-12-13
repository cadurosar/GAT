import time, sys, os
import scipy.sparse as sp
import numpy as np
#import tensorflow as tf
import argparse as ap
from multiprocessing import Pool
import torch

from models import MLP #SpGAT, SpGCT, SpGCTS, SpGCN, SpTAGCN,
from utils import process
import hashlib

parser = ap.ArgumentParser(description='Run 100 times and return mean acc +- std')

parser.add_argument('--dataset', '-d', default='cora')
parser.add_argument('--model', '-m', default='gct')
parser.add_argument('--patience', '-p', type=int, default=100)
parser.add_argument('--hiddens', '-hu', nargs='+', type=int, default=[8])
parser.add_argument('--nheads', '-n', nargs='+', type=int, default=[1, 1])
parser.add_argument('--savebest', '-b', type=bool, default=True)
parser.add_argument('--activation', '-a', default='relu')
parser.add_argument('--intraactivation', '-ia', default='None')
parser.add_argument('--indropout', '-id', type=float, default=0.5)
parser.add_argument('--rightdropout', '-rd', type=float, default=0.5)
parser.add_argument('--leftdropout', '-ld', type=float, default=0.5)
parser.add_argument('--snorm', '-s', default='softmax')
parser.add_argument('--nruns', '-r', type=int, default=100)
#parser.add_argument('--nthreads', '-t', type=int, default=25)
parser.add_argument('--usebias', '-ub', type=bool, default=True)
parser.add_argument('--verbose', '-v', type=bool, default=False)
#parser.add_argument('--std_init', '-std', default='None')

parser.add_argument('--cov', '-c', type=float, default=0.0)

args = parser.parse_args()
print(args)

checkpt_file = 'pre_trained/runner.ckpt'
dataset = args.dataset
if args.model == 'gct':
    model = SpGCT
elif args.model == 'gcts':
    model = SpGCTS
elif args.model == 'gcn':
    model = SpGCN
elif args.model == 'gat':
    model = SpGAT
elif args.model == 'tagcn':
    model = SpTAGCN
elif args.model == 'mlp':
    model = MLP
else:
    sys.exit(args.model + ' model unknown')

# training params
batch_size = 1
nb_epochs = 100000
patience = args.patience
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
hid_units = args.hiddens # numbers of hidden units per each attention head in each layer
n_heads = args.nheads # additional entry for the output layer
residual = False
save_best = args.savebest
if args.activation == 'elu':
    nonlinearity = torch.nn.functional.elu
elif args.activation == 'relu':
    nonlinearity = torch.nn.functional.relu
else:
    sys.exit(args.activation + ' activation unknown')
if args.intraactivation == 'None':
    nonlinearity2 = None
elif args.intraactivation == 'elu':
    nonlinearity2 = torch.nn.functional.elu
elif args.intraactivation == 'relu':
    nonlinearity2 = torch.nn.functional.relu
else:
    sys.exit(args.intraactivation + ' activation unknown')
attn_drop_value = args.leftdropout
ffd_drop_value = args.indropout
intra_drop_value = args.rightdropout
if args.snorm == 'softmax':
    scheme_norm = tf.sparse_softmax
    #scheme_norm = lambda x: tf.sparse_transpose(tf.sparse_softmax(tf.sparse_transpose(x)))
elif args.snorm == 'sum':
    def sparse_norm(x):
        rsum = tf.sparse_reduce_sum_sparse(x, axis=0, keep_dims=True)
        tf.SparseTensor(indices=x.indices,
                    values=x.values / rsum.values,
                    dense_shape=x.dense_shape)
        return x
    scheme_norm = sparse_norm
elif args.snorm == 'None':
    scheme_norm = None
else:
    sys.exit(args.snorm + ' snorm unknown')
scheme_init_std = None
use_bias = args.usebias

# whether to use thresholded covar adjacency matrix
use_covar = args.cov > 0
ratio_kept = args.cov

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

# using covar adj instead of given one
if use_covar:
    adj = np.cov(features)
    nb_kept = int(ratio_kept * adj.shape[0] * adj.shape[1])
    threshold = adj.flatten()[-nb_kept]
    adj[adj<threshold] = 0.0
    adj = sp.csr_matrix(adj)
###

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.preprocess_adj_bias(adj) if args.model == 'gat' else process.preprocess_adj(adj)
nnz = len(biases[1])

def run_once(run_id):

    local_model = model(nb_classes, nb_nodes, False,
                                attn_drop_value, ffd_drop_value, nnz, in_sz=ft_size,
                                bias_mat=biases,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity,
                                intra_drop=intra_drop_value, intra_activation=nonlinearity2,
                                scheme_norm=scheme_norm, scheme_init_std=scheme_init_std,
                                use_bias=use_bias)
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    local_model.apply(init_weights)

    print("starting")
    local_model.cuda()
    optimizer = torch.optim.Adam(local_model.parameters(),lr=lr,amsgrad=True)#,weight_decay=l2_coef)

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0
    bbias = biases

    for epoch in range(nb_epochs):
        tr_step = 0
        tr_size = features.shape[0]
        local_model.train()

        while tr_step * batch_size < tr_size:
            ftr_in = torch.cuda.FloatTensor(features[tr_step*batch_size:(tr_step+1)*batch_size])
            lbl_in = torch.cuda.LongTensor(y_train[tr_step*batch_size:(tr_step+1)*batch_size])
            msk_in = torch.cuda.FloatTensor(train_mask[tr_step*batch_size:(tr_step+1)*batch_size].astype(np.float32))
            logits = local_model(ftr_in)
            log_resh = logits.view(-1,nb_classes)
            lab_resh = lbl_in.view(-1, nb_classes)
            msk_resh = msk_in.view(-1)
            values = [(v**2).sum()/2 for v in local_model.parameters()]
            lossL2 = 0
            for a in values:
                lossL2 += a
            loss_value_tr = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh) + l2_coef*lossL2
            acc_tr = model.masked_accuracy(log_resh, lab_resh, msk_resh)
            loss_value_tr.backward()
            optimizer.step()
            train_loss_avg += loss_value_tr
            train_acc_avg += acc_tr
            tr_step += 1

        vl_step = 0
        vl_size = features.shape[0]
        local_model.eval()
        while vl_step * batch_size < vl_size:
            with torch.no_grad():
                ftr_in = torch.cuda.FloatTensor(features[vl_step*batch_size:(vl_step+1)*batch_size])
                lbl_in = torch.cuda.LongTensor(y_val[vl_step*batch_size:(vl_step+1)*batch_size])
                msk_in = torch.cuda.FloatTensor(val_mask[vl_step*batch_size:(vl_step+1)*batch_size].astype(np.float32))
                logits = local_model(ftr_in)
                log_resh = logits.view(-1,nb_classes)
                lab_resh = lbl_in.view(-1, nb_classes)
                msk_resh = msk_in.view(-1)
                loss_value_vl = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
                acc_vl = model.masked_accuracy(log_resh, lab_resh, msk_resh)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

        if args.verbose:
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                (train_loss_avg/tr_step, train_acc_avg/tr_step,
                val_loss_avg/vl_step, val_acc_avg/vl_step))

        if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
            if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                vacc_early_model = val_acc_avg/vl_step
                vlss_early_model = val_loss_avg/vl_step
                import copy
                final_model = copy.deepcopy(local_model)
            vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
            vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)

                break
        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

    ts_size = features.shape[0]
    ts_step = 0
    ts_loss = 0.0
    ts_acc = 0.0
    local_model.eval()
    while ts_step * batch_size < ts_size:
        with torch.no_grad():
            ftr_in = torch.cuda.FloatTensor(features[ts_step*batch_size:(ts_step+1)*batch_size])
            lbl_in = torch.cuda.LongTensor(y_test[ts_step*batch_size:(ts_step+1)*batch_size])
            msk_in = torch.cuda.FloatTensor(test_mask[ts_step*batch_size:(ts_step+1)*batch_size].astype(np.float32))
            logits = final_model(ftr_in)
            log_resh = logits.view(-1,nb_classes)
            lab_resh = lbl_in.view(-1, nb_classes)
            msk_resh = msk_in.view(-1)
            loss_value_ts = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            acc_ts = model.masked_accuracy(log_resh, lab_resh, msk_resh)

            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

    print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
    return ts_acc/ts_step

#pool = Pool()
#res_list = pool.map(run_once, range(args.nruns))
res_list = []
for i in range(args.nruns):
  res_list.append(run_once(i))

res = 'Test accuracy ' + str(args.nruns) + ' runs: ' + str(np.mean(res_list)) + ' +- ' + str(np.std(res_list))
print(res)

file = open(args.model + 'logger', 'a')
file.write(str(args) + '\n' + res + '\n')
file.close()
