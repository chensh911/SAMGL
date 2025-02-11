import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.ops import edge_softmax
import dgl.function as fn


class RelationGraphConv(nn.Module):

    def __init__(self, in_feats: tuple, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        """

        Parameters
        ----------
        in_feats : pair of ints
            Input feature size.
        out_feats : int
            Output feature size.
        num_heads : int
            Number of heads in Multi-Head Attention.
        dropout : float, optional
            Dropout rate, defaults: 0.
        negative_slope : float, optional
            Negative slope rate, defaults: 0.2.
        """
        super(RelationGraphConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tuple, dst_node_transformation_weight: nn.Parameter,
                src_node_transformation_weight: nn.Parameter, relation_attention_weight: nn.Parameter):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight: Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight: Parameter (input_src_dim, n_heads * hidden_dim)
        relation_attention_weight: Parameter (n_heads, 2 * hidden_dim)

        Returns
        -------
        torch.Tensor, shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        graph = graph.local_var()
        # Tensor, (N_src, input_src_dim)
        feat_src = self.dropout(feat[0])
        # Tensor, (N_dst, input_dst_dim)
        feat_dst = self.dropout(feat[1])
        # Tensor, (N_src, n_heads, hidden_dim) -> (N_src, input_src_dim) * (input_src_dim, n_heads * hidden_dim)
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (N_dst, n_heads, hidden_dim) -> (N_dst, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * relation_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * relation_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        # (N_src, n_heads, hidden_dim), (N_src, n_heads, 1)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        # (N_dst, n_heads, 1)
        graph.dstdata.update({'e_dst': e_dst})
        # compute edge attention, e_src and e_dst are a_src * Wh_src and a_dst * Wh_dst respectively.
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        # shape (edges_num, heads, 1)
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)

        # 1) one way: send and receive implementation
        # graph.send_and_recv(graph.edges(), self.gcn_message, self.gcn_reduce)
        # 2) another way: message passing
        # graph.update_all(self.gcn_message, self.gcn_reduce)
        # 3) another way: faster implementation using the fn, function
        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'ft'))
        # (N_dst, n_heads * hidden_dim), reshape (N_dst, n_heads, hidden_dim)
        dst_features = graph.dstdata.pop('ft').reshape(-1, self._num_heads * self._out_feats)

        dst_features = self.relu(dst_features)

        return dst_features

    @staticmethod
    def gcn_message(edges: dgl.udf.EdgeBatch):
        # (E, n_heads, hidden_dim), -> (E, n_heads, hidden_dim) * (E, heads, 1), E is the number of edges
        return {'msg': edges.src['ft'] * edges.data['a']}

    @staticmethod
    def gcn_reduce(nodes: dgl.udf.NodeBatch):
        # Tensor, (N, neighbors, n_heads, hidden_dim) -> (N, n_heads, hidden_dim)
        return {'ft': torch.sum(nodes.mailbox['msg'], 1)}
