from graphsaint.globals import *
import numpy as np
import scipy.sparse
import abc
import time
import math
import pdb
from math import ceil
import graphsaint.cython_sampler as cy
import copy

class graph_sampler:
    __metaclass__ = abc.ABCMeta
    def __init__(self,adj_train,node_train,size_subgraph,args_preproc):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    @abc.abstractmethod
    def preproc(self,**kwargs):
        pass

    def par_sample(self,stage,**kwargs):
        return self.cy_sampler.par_sample()


class rw_sampling(graph_sampler):
    def __init__(self,adj_train,node_train,size_subgraph,size_root,size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root*size_depth
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.RW(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.size_root,self.size_depth)
    def preproc(self,**kwargs):
        pass


class brw_sampling(graph_sampler):
    def __init__(self,adj_train,node_train,size_subgraph,size_root,size_depth,node_user,adj_bias_data):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root*size_depth
        self.node_user = node_user
        self.adj_bias_data = adj_bias_data
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.BRW(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.size_root,self.size_depth,self.node_user,self.adj_bias_data)
    
    def preproc(self,**kwargs):
        pass

class fbrw_sampling(graph_sampler):
    def __init__(self,adj_train,node_train,size_subgraph,size_root,size_depth,node_user):
        self.size_root = size_root
        self.size_depth = size_depth
        self.node_user = np.unique(node_user).astype(np.int32)
        size_subgraph = size_root*size_depth
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.FBRW(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.size_root,self.size_depth,self.node_user)
    def preproc(self,**kwargs):
        pass



class edge_sampling(graph_sampler):
    def __init__(self, edge_prob, delta, adj_train, node_train,num_edges_subgraph):
        """
        num_edges_subgraph: specify the size of subgraph by the edge budget. NOTE: other samplers specify node budget.
        """
        self.num_edges_subgraph = num_edges_subgraph
        self.delta = copy.deepcopy(delta)
        self.edge_prob = copy.deepcopy(edge_prob)
        self.size_subgraph = num_edges_subgraph*2       # this may not be true in many cases. But for now just use this.
        self.deg_train = np.array(adj_train.sum(1)).flatten()
        self.adj_train_norm = scipy.sparse.dia_matrix((1/self.deg_train,0),shape=adj_train.shape).dot(adj_train)
        super().__init__(adj_train,node_train,self.size_subgraph,dict())
        #self.cy_sampler = cy.Edge(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
        #    NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.edge_prob_tri.row,self.edge_prob_tri.col,self.edge_prob_tri.data)
        self.cy_sampler = cy.Edge2(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.edge_prob_tri.row,self.edge_prob_tri.col,self.edge_prob_tri.data.cumsum(),self.num_edges_subgraph)
    def preproc(self,**kwargs):
        
        # print('zzq edge_sampling preproc reduce variance')
        # self.edge_prob = scipy.sparse.csr_matrix((np.zeros(self.adj_train.data.size),\
        #         self.adj_train.indices,self.adj_train.indptr),shape=self.adj_train.shape)
        # self.edge_prob.data[:] = self.adj_train_norm.data[:]
        # _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        # self.edge_prob.data += _adj_trans.data      # P_e \propto a_{u,v} + a_{v,u}
        # # print('zzq edge_prob',self.edge_prob.data[:10])
        # for target_node in [1005,1094]:
        #     start_indptr = self.edge_prob.indptr[target_node]
        #     end_indptr = self.edge_prob.indptr[target_node+1]
        #     node_i_neighbors = self.edge_prob.indices[start_indptr:end_indptr]
        #     print('zzq node {}'.format(target_node))
        #     print('zzq edge_prob')
        #     for i in self.edge_prob.data[start_indptr:end_indptr]:
        #         print('{:.2f}'.format(i),end=' ')
        #     print()
        # self.edge_prob.data *= 2*self.num_edges_subgraph/self.edge_prob.data.sum()  
        # # now edge_prob is a symmetric matrix, we only keep the upper triangle part, since adj is assumed to be undirected.
        # self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32)  # NOTE: in coo format

        # print('zzq edge_sampling preproc uniform sampling')
        # self.edge_prob = scipy.sparse.csr_matrix((copy.deepcopy(self.adj_train.data).astype(np.float64),\
        #         self.adj_train.indices,self.adj_train.indptr),shape=self.adj_train.shape)

        # for target_node in [1005,1094]:
        #     start_indptr = self.edge_prob.indptr[target_node]
        #     end_indptr = self.edge_prob.indptr[target_node+1]
        #     node_i_neighbors = self.edge_prob.indices[start_indptr:end_indptr]
        #     print('zzq node {}'.format(target_node))
        #     print('zzq edge_prob')
        #     for i in self.edge_prob.data[start_indptr:end_indptr]:
        #         print('{:.2f}'.format(i),end=' ')
        #     print()
        # self.edge_prob.data *= 2*self.num_edges_subgraph/self.edge_prob.data.sum() 

        # # now edge_prob is a symmetric matrix, we only keep the upper triangle part, since adj is assumed to be undirected.
        # self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32)  # NOTE: in coo format


        print('zzq combine sampling')

        # print('edge_prob in sampling')
        # for i in self.edge_prob.data[:10]:
        #     print('{:.2f}'.format(i),end=' ')
        # print()
        
        # self.edge_prob.data += self.delta.data

        # print('delta in sampling')
        # for i in self.delta.data[:10]:
        #     print('{:.2f}'.format(i),end=' ')
        # print()

        # self.edge_prob.data[self.edge_prob.data <= 0.0] = 1e-4

        self.edge_prob.data[self.delta.data<0.0] = 1e-4

        for target_node in [1005,1094]:
            start_indptr = self.edge_prob.indptr[target_node]
            end_indptr = self.edge_prob.indptr[target_node+1]
            node_i_neighbors = self.edge_prob.indices[start_indptr:end_indptr]
            print('zzq node {}'.format(target_node))
            print('zzq delta')
            for i in self.delta.data[start_indptr:end_indptr]:
                print('{:.2f}'.format(i),end=' ')
            print()
            print('zzq edge_prob')
            for i in self.edge_prob.data[start_indptr:end_indptr]:
                print('{:.2f}'.format(i),end=' ')
            print()
            print()

        edge_prob_copy = copy.deepcopy(self.edge_prob)
        edge_prob_copy.data *= 2*self.num_edges_subgraph/edge_prob_copy.data.sum() 

        # now edge_prob is a symmetric matrix, we only keep the upper triangle part, since adj is assumed to be undirected.
        self.edge_prob_tri = scipy.sparse.triu(edge_prob_copy).astype(np.float32)  # NOTE: in coo format

class mrw_sampling(graph_sampler):

    def __init__(self,adj_train,node_train,size_subgraph,size_frontier,max_deg=10000):
        self.p_dist = None
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.size_frontier = size_frontier
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'MRW'
        self.max_deg = int(max_deg)
        self.cy_sampler = cy.MRW(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.p_dist,self.max_deg,self.size_frontier,self.size_subgraph)

    def preproc(self,**kwargs):
        _adj_hop = self.adj_train
        self.p_dist = np.array([_adj_hop.data[_adj_hop.indptr[v]:_adj_hop.indptr[v+1]].sum() for v in range(_adj_hop.shape[0])], dtype=np.int32)




class node_sampling(graph_sampler):
    
    def __init__(self,adj_train,node_train,size_subgraph):
        self.p_dist = np.zeros(len(node_train))
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.Node(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.p_dist,self.size_subgraph)

    def preproc(self,**kwargs):
        _p_dist = np.array([self.adj_train.data[self.adj_train.indptr[v]:self.adj_train.indptr[v+1]].sum() for v in self.node_train], dtype=np.int64)
        self.p_dist = _p_dist.cumsum()
        print('zzq preproc len(_p_dist)',len(_p_dist),self.adj_train.shape[0])
        if self.p_dist[-1] > 2**31-1:
            print('warning: total deg exceeds 2**31')
            self.p_dist = self.p_dist.astype(np.float64)
            self.p_dist /= self.p_dist[-1]/(2**31-1)
        self.p_dist = self.p_dist.astype(np.int32)


class node_partial_edge_sampling(graph_sampler):

    def __init__(self,adj_train,node_train,size_subgraph,est_adj_train):
        self.p_dist = np.zeros(len(node_train))
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.Node_Partial_Edge(adj_train.indptr, adj_train.indices, self.node_train,
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC,self.p_dist,
            est_adj_train.indptr, est_adj_train.indices, est_adj_train.data,
            self.size_subgraph)

    def preproc(self,**kwargs):
        _p_dist = np.array([self.adj_train.data[self.adj_train.indptr[v]:self.adj_train.indptr[v+1]].sum() for v in self.node_train], dtype=np.int64)
        self.p_dist = _p_dist.cumsum()
        print('zzq node_partial_edge_sampling')
        if self.p_dist[-1] > 2**31-1:
            print('warning: total deg exceeds 2**31')
            self.p_dist = self.p_dist.astype(np.float64)
            self.p_dist /= self.p_dist[-1]/(2**31-1)
        self.p_dist = self.p_dist.astype(np.int32)
        

class full_batch_sampling(graph_sampler):
    
    def __init__(self,adj_train,node_train,size_subgraph):
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.FullBatch(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
            NUM_PAR_SAMPLER,SAMPLES_PER_PROC)

