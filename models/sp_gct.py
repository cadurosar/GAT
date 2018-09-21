import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class SpGCT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop, nnz,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu,
            intra_drop=None, intra_activation=None, scheme_norm=None, scheme_init_std=None,
            residual=False):
        attns = []
        with tf.variable_scope('level_0'):
            for _ in range(n_heads[0]):
              with tf.variable_scope('head_' + str(_)):
                attns.append(layers.sp_cttn_head(inputs,
                    adj_mat=bias_mat, nnz=nnz,
                    out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            with tf.variable_scope('level_' + str(i)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    with tf.variable_scope('head_' + str(_)):
                      attns.append(layers.sp_cttn_head(h_1,  
                        adj_mat=bias_mat, nnz=nnz,
                        out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                        in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
                h_1 = tf.concat(attns, axis=-1)
        out = []
        with tf.variable_scope('level_' + str(len(hid_units))):
            for i in range(n_heads[-1]):
                with tf.variable_scope('head_' + str(i)):
                  out.append(layers.sp_cttn_head(h_1, adj_mat=bias_mat, nnz=nnz,
                    out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits

(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=False, scheme_norm=None,
                 scheme_init_std=None):