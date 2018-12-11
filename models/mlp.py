import torch
import utils
from models.base_gattn import BaseGAttN

class MLP(BaseGAttN):
    def __init__(self,nb_classes, nb_nodes, training, attn_drop, ffd_drop, nnz,in_sz,
            bias_mat, hid_units, n_heads, activation=torch.nn.functional.elu,
            intra_drop=None, intra_activation=None, scheme_norm=None, scheme_init_std=None,
            residual=False, use_bias=True):
        super(MLP, self).__init__()
        self.n_heads = n_heads
        self.attns = []
        for _ in range(n_heads[0]):
            self.attns.append(utils.layers.mlp_head(in_sz=in_sz,
                    adj_mat=bias_mat, nnz=nnz,
                    out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                    intra_drop=intra_drop, intra_activation=intra_activation,
                    scheme_norm=scheme_norm, scheme_init_std=scheme_init_std, use_bias=use_bias))
        self.attns = utils.ListModule(*self.attns)
        self.intermediate_attns = []
        for i in range(1, len(hid_units)):
            internal_attns = []
            for _ in range(n_heads[i]):
                internal_attns.append(utils.layers.mlp_head(in_sz=hid_units[i-1],  
                adj_mat=bias_mat, nnz=nnz,
                out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual,
                intra_drop=intra_drop, intra_activation=intra_activation,
                scheme_norm=scheme_norm, scheme_init_std=scheme_init_std, use_bias=use_bias))
            internal_attns = utils.ListModule(*internal_attns)
            self.intermediate_attns.append(internal_attns)
        self.intermediate_attns = utils.ListModule(*self.intermediate_attns)

        self.out = []
        for i in range(n_heads[-1]):
            self.out.append(utils.layers.mlp_head(in_sz=hid_units[-1],adj_mat=bias_mat, nnz=nnz,
            out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
            in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
            intra_drop=intra_drop, intra_activation=intra_activation,
            scheme_norm=scheme_norm, scheme_init_std=scheme_init_std, use_bias=use_bias))
        self.out = utils.ListModule(*self.out)
        self.nb_classes = nb_classes
    
    def forward(self,input):
        input = torch.transpose(input,1,2)
        outs = []
        for a in self.attns:
            outs.append(a(input))
        h_1 = torch.cat(outs, dim=1)
        for layer in self.intermediate_attns:
            outs = []
            for a in layer:
                outs.append(a(h_1))
            h_1 = torch.cat(outs,dim=1)
        out = torch.cuda.FloatTensor(h_1.size(0),self.nb_classes,h_1.size(2))
        out.zero_()
        for a in self.out:
            out += a(h_1)
        logits = out / self.n_heads[-1]
        logits = torch.transpose(logits,1,2)
        return logits
