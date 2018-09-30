import time, sys, os
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse as ap
from multiprocessing import Pool

from models import SpGAT, SpGCT, SpGCTS, SpGCN
from utils import process
import hashlib

parser = ap.ArgumentParser(description='Run 100 times and return mean acc +- std')

parser.add_argument('--dataset', '-d', default='cora')
parser.add_argument('--model', '-m', default='gct')
parser.add_argument('--patience', '-p', type=int, default=100)
parser.add_argument('--hiddens', '-hu', nargs='+', type=int, default=[8])
parser.add_argument('--nheads', '-n', nargs='+', type=int, default=[8, 1])
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

args = parser.parse_args()
print(args, flush=True)

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
else:
    sys.exit(args.model + ' model unknown')

# training params
batch_size = 1
nb_epochs = 100000
patience = args.patience
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = args.hiddens # numbers of hidden units per each attention head in each layer
n_heads = args.nheads # additional entry for the output layer
residual = False
save_best = args.savebest
if args.activation == 'elu':
    nonlinearity = tf.nn.elu
elif args.activation == 'relu':
    nonlinearity = tf.nn.relu
else:
    sys.exit(args.activation + ' activation unknown')
if args.intraactivation == 'None':
    nonlinearity2 = None
elif args.intraactivation == 'elu':
    nonlinearity2 = tf.nn.elu
elif args.intraactivation == 'relu':
    nonlinearity2 = tf.nn.relu
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

"""
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
"""

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

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
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            intra_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                    attn_drop, ffd_drop, nnz,
                                    bias_mat=bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity,
                                    intra_drop=intra_drop, intra_activation=nonlinearity2,
                                    scheme_norm=scheme_norm, scheme_init_std=scheme_init_std,
                                    use_bias=use_bias)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        train_op = model.training(loss, lr, l2_coef)

        if save_best:
            saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(init_op)

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]

                while tr_step * batch_size < tr_size:
                    bbias = biases

                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: bbias,
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: attn_drop_value, ffd_drop: ffd_drop_value,
                            intra_drop: intra_drop_value})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]

                while vl_step * batch_size < vl_size:
                    bbias = biases
                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                            bias_in: bbias,
                            lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                            msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0, intra_drop: 0.0})
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
                        if save_best:
                            saver.save(sess, checkpt_file + '_' + hashlib.md5(str(args).encode()).hexdigest() + '_proc' + str(run_id))
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

            if save_best:
                saver.restore(sess, checkpt_file + '_' + hashlib.md5(str(args).encode()).hexdigest() +  '_proc' + str(run_id))

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                bbias = biases
                loss_value_ts, acc_ts = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0, intra_drop:0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

            print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

            sess.close()
            os.system('rm ' + checkpt_file + '_' + hashlib.md5(str(args).encode()).hexdigest() +  '_proc' + str(run_id) + '*')
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
