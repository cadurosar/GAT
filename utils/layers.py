import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

# attention (Velickovic et al.)
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False, use_bias=True):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        if use_bias:
            ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=None, scheme_norm=False,
                 scheme_init_std=None):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# neural contraction (Vialatte et al.)
def sp_cttn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=None, scheme_norm=tf.sparse_softmax,
                 scheme_init_std=None):
    if intra_drop is None:
        intra_drop = in_drop
    
    with tf.name_scope('sp_contraction'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # right operand SXW
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        if not(intra_activation is None):
            seq_fts = intra_activation(seq_fts)
        if intra_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - intra_drop)

        # left operand SXW
        initializer = None if scheme_init_std is None else tf.truncated_normal_initializer(0.0,scheme_init_std)
        scheme_kernel = tf.get_variable('scheme_kernel', (nnz,),
                                        initializer=initializer,
                                        trainable=True)
        scheme = tf.SparseTensor(indices = adj_mat.indices,
            values = scheme_kernel,
            dense_shape = adj_mat.dense_shape)
        scheme = tf.sparse_add(scheme, adj_mat)
        if not(scheme_norm is None):
            scheme = scheme_norm(scheme)
        if coef_drop != 0.0:
            scheme = tf.SparseTensor(indices=scheme.indices,
                    values=tf.nn.dropout(scheme.values, 1.0 - coef_drop),
                    dense_shape=scheme.dense_shape)
        scheme = tf.sparse_reshape(scheme, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(scheme, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])

        # bias
        if use_bias:
            ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        # activation
        return activation(ret)

# original graph convolution (Kipf et al.)
def sp_gcn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=None, scheme_norm=None,
                 scheme_init_std=None):
    if intra_drop is None:
        intra_drop = in_drop
    
    with tf.name_scope('sp_gcn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # right operand SXW
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        if not(intra_activation is None):
            seq_fts = intra_activation(seq_fts)
        if intra_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - intra_drop)

        # left operand SXW        
        scheme = adj_mat
        if not(scheme_norm is None):
            scheme = scheme_norm(scheme)
        if coef_drop != 0.0:
            scheme = tf.SparseTensor(indices=scheme.indices,
                    values=tf.nn.dropout(scheme.values, 1.0 - coef_drop),
                    dense_shape=scheme.dense_shape)
        scheme = tf.sparse_reshape(scheme, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(scheme, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])

        # bias
        if use_bias:
            ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        # activation
        return activation(ret)

# topology adaptative graph convolution (Du et al.)
def sp_tagcn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=None, scheme_norm=None,
                 scheme_init_std=None, K=2): #K is the polynomial order
    if intra_drop is None:
        intra_drop = in_drop
    
    with tf.name_scope('sp_gcn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # preprocessing S
        scheme = adj_mat
        if not(scheme_norm is None):
            scheme = scheme_norm(scheme)
        if coef_drop != 0.0:
            scheme = tf.SparseTensor(indices=scheme.indices,
                    values=tf.nn.dropout(scheme.values, 1.0 - coef_drop),
                    dense_shape=scheme.dense_shape)
        scheme = tf.sparse_reshape(scheme, [nb_nodes, nb_nodes])

        vals = None
        assert K > 0
        for k in range(1, K+1):

            # right operand SXW
            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
            if not(intra_activation is None):
                seq_fts = intra_activation(seq_fts)
            if intra_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - intra_drop)
            seq_fts = tf.squeeze(seq_fts)

            # left operand SXW
            for ik in range(k):
                seq_fts = tf.sparse_tensor_dense_matmul(scheme, seq_fts)

            # sum
            if vals is None:
                vals = seq_fts
            else:
                vals = vals + seq_fts

        # shape
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])

        # bias
        if use_bias:
            ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        # activation
        return activation(ret)

# regular fully connected layer without using the graph
import torch
class mlp_head(torch.nn.Module):
    def __init__(seq, out_sz, adj_mat=None, activation=torch.nn.ReLU, nb_nodes=None, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=None, scheme_norm=None,
                 scheme_init_std=None):

    def forward(input)

def mlp_head(seq, out_sz, adj_mat=None, activation=tf.nn.relu, nb_nodes=None, in_drop=0.0, coef_drop=0.0, residual=False,
                 nnz=None, use_bias=True, intra_drop=None, intra_activation=None, scheme_norm=None,
                 scheme_init_std=None):
    if intra_drop is None:
        intra_drop = in_drop
    
    with tf.name_scope('mlp'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # operation XW
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        if not(intra_activation is None):
            seq_fts = intra_activation(seq_fts)
        #if intra_drop != 0.0:
        #    seq_fts = tf.nn.dropout(seq_fts, 1.0 - intra_drop)

        # bias
        if use_bias:
            ret = tf.contrib.layers.bias_add(seq_fts)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        # activation
        return activation(ret)
        
