import numpy as np
import torch
import random

from itertools import combinations 
import networkx as nx

EPS = 1e-40

class Pairwise_Graph():
    def __init__(self, dct_node_idx_to_alphabet, edges, random_graph_params=None):
        if random_graph_params is not None:
            assert edges is None
            edges = get_random_edges(n_vars=len(dct_node_idx_to_alphabet), **random_graph_params)

        self.G = self.build_structure(edges, dct_node_idx_to_alphabet)

    # build structure of graph using networkx functionality
    def build_structure(self, edges, dct_node_idx_to_alphabet):
        def validate_edges(edges):
            for e in edges:
                assert type(e) == type((0,0))
                assert len(e) == 2
                assert e[0] in dct_node_idx_to_alphabet and e[1] in dct_node_idx_to_alphabet
                assert (e[1],e[0]) not in edges
            return

        # Warning!!!
        # assumption of this class is that this order of edges (keys) is for ever maintained
        # any edge (a,b) with a<b is at idx i and edge (b,a) has to be at idx i+1
        def get_bidirectional_and_ordered_edges(edges):
            bidir = []
            for i, e in enumerate(edges): 
                bidir.append(e)
                bidir.append((e[1],e[0]))
            return bidir
        
        # add all nodes first
        assert list(dct_node_idx_to_alphabet.keys()) == list(range(len(list(dct_node_idx_to_alphabet.keys()))))
        self.nodes = {v: {'alphabet': alph} for v, alph in dct_node_idx_to_alphabet.items()}

        # validate and add edges
        edges = sorted(edges)

        validate_edges(edges)

        # add shortcut to undirected edges for user to push in potentials
        self.undir_edges = edges

        # but work with bidir edges under the hood for message passing
        self.edges = {e: {} for e in get_bidirectional_and_ordered_edges(edges)}

        # make nbrs dct
        self.nbrs = {v: [] for v in self.nodes}
        for e in self.edges: 
            self.nbrs[e[0]].append(e[1])
            self.nbrs[e[1]].append(e[0])
        return

    def nbrs(self, idx): return self.nbrs[idx]

    # build a networkx graph to draw
    def draw(self, with_labels=True, node_size=1000, font_color='white', font_size=20):
        G=nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.undir_edges)
        nx.draw(G, pos=nx.kamada_kawai_layout(G), with_labels=with_labels, node_size=node_size, font_color=font_color, font_size=font_size)
        return 

class Pairwise_BP_Computer_Parallel():
    def __init__(self, graph, device):
        # will use the order of edges in self.G.edges as the reference point for mx ops
        # note that self.G.edges has every edge in forward and backward directions
        # [(s_1,t_1), ..., (s_|E|,t_|E|)]
        # similarly, will use order of nodes in self.G.nodes as a fixed reference
        # nodes referred to as v_1, ..., v_n
        self.G = graph

        self.device = device

        # allow nodes to have varying alphabet lengths
        # consecutive elements from 0 onwards
        # ops scale to bigger alphabet size
        self.alphabet = list(range(max([len(self.G.nodes[i]['alphabet']) for i in self.G.nodes])))
        
        # mx ops follow Reid M. Bizler's work
        # https://vtechworks.lib.vt.edu/bitstream/handle/10919/83228/Bixler_RM_T_2018.pdf
        
        ### BP items
        ## indexing mxs

        # Note! for fast building of indexing matrices, we assume:
        # (1) variables go from 0 to len(variables), consecutively
        # (2) composite edge placement in G.edges is never altered

        # T_ij = 1 if for ith edge (s_i,t_i), t_i is node v_j
        self.T = self.get_TO_matrix().to(self.device)
        assert self.T.shape == (len(self.G.edges), len(self.G.nodes))
        # F_ij = 1 if for ith edge (s_i,t_i), s_i is node v_j
        self.F = self.get_FROM_matrix().to(self.device)
        assert self.F.shape == (len(self.G.edges), len(self.G.nodes))
        # R_ij = 1 if the reverse of the ith edge, (say (v_a,v_b) with reverse (v_b,v_a)) is the jth edge 
        self.R = self.get_edge_R_matrix().to(self.device)
        assert self.R.shape == (len(self.G.edges), len(self.G.edges))
        return

    def get_TO_matrix(self):
        # T_ij = 1 if for ith edge (s_i,t_i), t_i is node v_j
        idxs = [tuple(range(len(self.G.edges))), tuple([e[1] for e in self.G.edges])]
        T = torch.zeros(len(self.G.edges), len(self.G.nodes)).to(self.device)
        T[idxs[0], idxs[1]] = 1
        return T
    
    def get_FROM_matrix(self):
        # F_ij = 1 if for ith edge (s_i,t_i), s_i is node v_j
        idxs = [tuple(range(len(self.G.edges))), tuple([e[0] for e in self.G.edges])]
        F = torch.zeros(len(self.G.edges), len(self.G.nodes)).to(self.device)
        F[idxs[0], idxs[1]] = 1
        return F
    
    def get_edge_R_matrix(self):
        # R_ij = 1 if the reverse of the ith edge, (say (v_a,v_b) with reverse (v_b,v_a)) is the jth edge 
        # in our ordering we know if ith edge is (a,b) with a < b then i+1th edge is (b,a)
        R = torch.zeros(len(self.G.edges), len(self.G.edges)).to(self.device)
        idxs = [tuple(range(len(self.G.edges))), tuple([i+1 if i%2==0 else i-1 for i in range(len(self.G.edges))])]
        R[idxs[0], idxs[1]] = 1
        return R
    
    def get_node_log_pots(self, obs, dct_var_to_log_pot, uniform_alphabet):
        # provided user log pots for nodes
        if len(dct_var_to_log_pot) > 0:
            assert obs.shape[1] == len(dct_var_to_log_pot)

            if uniform_alphabet:
                log_pots = torch.stack([lp for lp in dct_var_to_log_pot.values()], dim=0)
            else:
                # pad with insignificant vals if alphabet not as long as longest
                padded_lps = []
                for v, lp in dct_var_to_log_pot.items():
                    pad = (torch.zeros(len(self.alphabet)-len(self.G.nodes[v]['alphabet'])).to(self.device)+EPS).log()
                    padded_lps.append(torch.cat([lp, pad], dim=0))

                log_pots = torch.stack(padded_lps, dim=0)

            assert log_pots.shape == (obs.shape[1], len(self.alphabet))
        
        # otherwise just make it all 1's by default
        else:
            log_pots = torch.ones(obs.shape[1], len(self.alphabet)).to(self.device)

        # start with log pots for every batch the same, per variable
        log_pots = log_pots.unsqueeze(0).repeat(obs.shape[0], 1, 1)
        assert log_pots.shape == (obs.shape[0], obs.shape[1], len(self.alphabet))

        # wherever we have observed values, force the log pots there
        for i in range(len(self.alphabet)):
            row = torch.zeros(len(self.alphabet)).to(self.device) + EPS
            row[i] = 1
            log_pots[obs==i] = row.log()

        return log_pots.transpose(2,1)
    
    def get_edge_log_pots(self, dct_edge_to_log_pot={}, uniform_alphabet=True):
        log_pots = []

        # pad with insignificant vals if alphabets not as long as longest
        def pad_edge_log_pot_2D(e, lp):
            if len(self.alphabet) > len(self.G.nodes[e[0]]['alphabet']):
                pad_rows = (torch.zeros(len(self.alphabet)-len(self.G.nodes[e[0]]['alphabet']), len(self.G.nodes[e[1]]['alphabet'])).to(self.device)+EPS).log()
                lp = torch.cat([lp, pad_rows], dim=0)
            
            if len(self.alphabet) > len(self.G.nodes[e[1]]['alphabet']):
                pad_cols = (torch.zeros(len(self.alphabet), len(self.alphabet)-len(self.G.nodes[e[1]]['alphabet'])).to(self.device)+EPS).log()
                lp = torch.cat([lp, pad_cols], dim=1)
            return lp

        # assumption on edges order exploited here (edge followed by its composite in order)
        # no bs needed as edge log pots are identical across batches, unlike node pots
        for e in dct_edge_to_log_pot.keys(): 
            assert (e[1],e[0]) not in dct_edge_to_log_pot
            assert dct_edge_to_log_pot[e].shape == (len(self.G.nodes[e[0]]['alphabet']), len(self.G.nodes[e[1]]['alphabet']))

            log_pots.append(pad_edge_log_pot_2D(e=(e[0],e[1]), lp=dct_edge_to_log_pot[e]))
            log_pots.append(pad_edge_log_pot_2D(e=(e[1],e[0]), lp=dct_edge_to_log_pot[e].transpose(1,0)))
            
        return torch.stack(log_pots, dim=2)
    
    # state, needs to be refreshed before every full BP computation cycle
    # no need to then build above mxs over and over again if G is unchanged in structure
    def reset_state(self, obs, psi=None, dct_var_to_log_pot={}, dct_edge_to_log_pot={}, uniform_alphabet=True):
        self.bs = obs.shape[0]
        
        # node log pots, combine the indiv nodes' log pots into a common mx
        self.phi = self.get_node_log_pots(obs, dct_var_to_log_pot, uniform_alphabet)
        assert self.phi.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
        
        # edge log pots, combine the indiv edges' log pots into a common mx
        if psi is None: 
            assert dct_edge_to_log_pot is not None
            self.psi = self.get_edge_log_pots(dct_edge_to_log_pot, uniform_alphabet)
        else:
            self.psi = psi
            
        assert self.psi.shape == (len(self.alphabet), len(self.alphabet), len(self.G.edges)) or \
                self.psi.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))

        # initial msgs mx
        self.M = torch.ones(self.bs, len(self.alphabet), len(self.G.edges)).to(self.device).log()
        assert self.M.shape == (self.bs, len(self.alphabet), len(self.G.edges))

        return

    def run_BP_till_convergence(self, max_iters, tol=1e-20, verbose=True):
        # check if state was set
        assert hasattr(self, 'M')
        assert hasattr(self, 'phi')
        assert hasattr(self, 'psi')

        for i in range(max_iters):     
            old_M = self.M.clone()
            assert old_M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            assert self.T.shape == (len(self.G.edges), len(self.G.nodes))
            
            B = self.get_univariate_marginals(msgs=old_M, normalize=True)
            assert B.shape == (self.bs, len(self.G.nodes), len(self.alphabet))
            
            B = B.transpose(2,1)
            assert B.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
            
            assert self.F.shape == (len(self.G.edges), len(self.G.nodes))
            assert self.F.transpose(1,0).shape == (len(self.G.nodes), len(self.G.edges))
            
            BF_t = torch.matmul(B, self.F.transpose(1,0))
            assert BF_t.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            assert old_M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            assert self.R.shape == (len(self.G.edges), len(self.G.edges))
            
            MR = torch.matmul(old_M, self.R)
            assert MR.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            BF_t_sub_MR = BF_t - MR
            assert BF_t_sub_MR.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            BF_t_sub_MR = BF_t_sub_MR.unsqueeze(2).repeat(1,1,len(self.alphabet),1)
            assert BF_t_sub_MR.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
            
            assert self.psi.shape == (len(self.alphabet), len(self.alphabet), len(self.G.edges)) or \
                self.psi.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
            
            pre_M = BF_t_sub_MR + self.psi
            assert pre_M.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
            
            self.M = torch.logsumexp(pre_M, dim=1)
            assert self.M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            if torch.all(torch.abs(self.M - old_M) < tol):
                if verbose: print(f'converged in {i} of {max_iters} iters')
                return
            
        if verbose: print(f'did not converge within {max_iters} iters')
        return
    
    def run_MAP_till_convergence(self, max_iters, tol=1e-20, verbose=True):
        # check if state was set
        assert hasattr(self, 'M')
        assert hasattr(self, 'phi')
        assert hasattr(self, 'psi')

        for i in range(max_iters):     
            old_M = self.M.clone()
            assert old_M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            assert self.T.shape == (len(self.G.edges), len(self.G.nodes))
            
            ## THIS IS THE ONLY DIFFERENCE -- idea, run B differently by calling marg differently then extract the rest as a method which takes in B
            B_unnorm = self.get_univariate_marginals(msgs=old_M, normalize=False)
            assert B_unnorm.shape == (self.bs, len(self.G.nodes), len(self.alphabet))
            
            def hard_samples_to_soft_samples(x):
                bs, n_vars = x.shape
                soft = torch.zeros(bs, n_vars, 2)
                soft[x == 0] = torch.FloatTensor([1,0])    
                soft[x == 1] = torch.FloatTensor([0,1])
                return soft.to(x.device)

            B_argmax = torch.argmax(B_unnorm, dim=2)
            assert B_argmax.shape == (self.bs, len(self.G.nodes))

            B = hard_samples_to_soft_samples(B_argmax)
            assert B.shape == (self.bs, len(self.G.nodes), len(self.alphabet))
            
            B = B.transpose(2,1)
            assert B.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
            
            assert self.F.shape == (len(self.G.edges), len(self.G.nodes))
            assert self.F.transpose(1,0).shape == (len(self.G.nodes), len(self.G.edges))
            
            BF_t = torch.matmul(B, self.F.transpose(1,0))
            assert BF_t.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            assert old_M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            assert self.R.shape == (len(self.G.edges), len(self.G.edges))
            
            MR = torch.matmul(old_M, self.R)
            assert MR.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            BF_t_sub_MR = BF_t - MR
            assert BF_t_sub_MR.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            BF_t_sub_MR = BF_t_sub_MR.unsqueeze(2).repeat(1,1,len(self.alphabet),1)
            assert BF_t_sub_MR.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
            
            assert self.psi.shape == (len(self.alphabet), len(self.alphabet), len(self.G.edges)) or \
                self.psi.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
            
            pre_M = BF_t_sub_MR + self.psi
            assert pre_M.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
            
            self.M = torch.logsumexp(pre_M, dim=1)
            assert self.M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
            
            if torch.all(torch.abs(self.M - old_M) < tol):
                if verbose: print(f'converged in {i} of {max_iters} iters')
                return
            
        if verbose: print(f'did not converge within {max_iters} iters')
        return hard_samples_to_soft_samples(B_argmax)

    # phi + MT (with or without normalization), BP wants normalized, MAP does not
    def get_univariate_marginals(self, msgs=None, normalize=True):
        # if msgs is None, read last recorded msgs after BP completion 
        # else, called from msg calculation procedure, with msgs provided
        if msgs is None: M = self.M
        else: M = msgs
            
        assert M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
        
        MT = torch.matmul(M, self.T)
        assert MT.shape == (self.bs, len(self.alphabet), len(self.G.nodes))

        assert self.phi.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
        B_unnorm = self.phi + MT
        assert B_unnorm.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
        
        if normalize:
            normalizer = torch.logsumexp(B_unnorm, dim=1, keepdim=True)
            assert normalizer.shape == (self.bs, 1, len(self.G.nodes))
            B = B_unnorm - normalizer
            assert B.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
            
        else:
            B = B_unnorm
        
        return B.transpose(2,1)


from tqdm import tqdm as tqdm

class Pairwise_Binary_Gibbs_Sampler(Pairwise_BP_Computer_Parallel):
    def __init__(self, graph, n_parallel, n_burnin, device, dct_edge_to_log_pot):
        super(Pairwise_Binary_Gibbs_Sampler, self).__init__(graph, device)
        self.alphabet = [0,1]
        self.n_parallel = n_parallel
        self.n_burnin = n_burnin
        self.device = device
        
        # edge log pots are permanent
        # node log pots and messages will be dealt with in each wave of sample creation
        self.psi = self.get_edge_log_pots(dct_edge_to_log_pot, uniform_alphabet=True).to(self.device)
        assert self.psi.shape == (len(self.alphabet), len(self.alphabet), len(self.G.edges))
        return
    
    def get_uar_samples(self, shape):
        x = torch.rand(*shape)
        return torch.bernoulli(x).to(self.device)
        
    def get_samples(self, n, tol_R, timeout):
        # collect and replace for burnin iters..., collect n, check R measure across processes, 
        # if R is satisfactory, return, else repeat from burnin...
        assert n % self.n_parallel == 0 and n >= self.n_parallel
        
        samples = self.get_uar_samples(shape=(self.n_parallel, len(self.G.nodes)))
        captures = []
        
        # set bs for the 1 iter of BP used to get samples
        self.bs = self.n_parallel
        
        tried = 0

        while True:
            for wave in tqdm(range(self.n_burnin + int(n / self.n_parallel))):
                samples = self.get_one_wave_of_samples(old_samples=samples)
                
                if wave >= self.n_burnin:
                    captures.append(samples)
                    
            tried += 1

            if tried == timeout:
                break
                
            if self.check_mixed(captures, tol_R): 
                break

            captures = []
            continue
                
        captures = torch.cat(captures, dim=0)
        assert captures.shape == (n, len(self.G.nodes))
        return captures
    
    def check_mixed(self, list_of_samples, tol_R):
        # each tensor is of shape n parallel processes x sample length
        # for each parallel process, get all corresponding samples together
        stacked_samples = torch.stack(list_of_samples, dim=1)
        assert len(stacked_samples.shape) == 3
        assert stacked_samples.shape[:2] == (self.n_parallel, len(list_of_samples))
        assert stacked_samples.shape[2] == len(self.G.nodes) or stacked_samples.shape[2] % len(self.G.nodes) == 0
        
        R = self.compute_mixing_misagreement_measure(stacked_samples)
        print(R)
        
        if abs(R - 1) < tol_R: 
            return True
        
        return False
    
    def compute_mixing_misagreement_measure(self, stacked_samples):
        K, M, d = stacked_samples.shape
        
        f_k = stacked_samples.mean(dim=1)
        assert f_k.shape == (K, d)
        
        f = f_k.mean(dim=0)
        assert f.shape == (d,)
        
        B = (M/(K-1)) * ((f_k-f)**2).sum(dim=0)
        assert B.shape == (d,)
        
        f_k = f_k.unsqueeze(1).repeat(1, M, 1)
        W = (1/K) * (1/(M - 1)) * ((stacked_samples - f_k)**2).sum(dim=1).sum(dim=0)
        assert W.shape == (d,)
        
        V = (((M-1)/M) * W) + ((1/M) * B)
        assert V.shape == (d,)
        
        # W is a vector and when conditioning on observations, could have hard 0's
        R = (V / (W+1e-40))**(0.5)
        assert R.shape == (d,)
        
        return R.mean()
        
    def get_one_wave_of_samples(self, old_samples, orders=None):
        n_threads = old_samples.shape[0]
        assert old_samples.shape[1] == len(self.G.nodes)
        
        # different order (kernel in Koller book) for each generating process
        if orders is None:
            orders = torch.stack([torch.randperm(len(self.G.nodes)) for i in range(n_threads)], dim=0).to(self.device)
        
        assert orders.shape[0] == old_samples.shape[0] and orders.shape[1] >= 1 and orders.shape[1] <= len(self.G.nodes)
        new_samples = old_samples.clone()
        
        for col_idx in range(orders.shape[1]):
            query_idxs = orders[:, col_idx]
            
            # current samples mx become the obs on graph, except query idxs which take val of -1
            new_samples[range(n_threads), query_idxs] = -1
            self.phi = self.get_node_log_pots(obs=new_samples, dct_var_to_log_pot={}, uniform_alphabet=True)
            assert self.phi.shape == (n_threads, len(self.alphabet), len(self.G.nodes))
            
            # reset the msgs matrix
            self.M = torch.ones(n_threads, len(self.alphabet), len(self.G.edges)).to(self.device).log()
            assert self.M.shape == (n_threads, len(self.alphabet), len(self.G.edges))
        
            # send one wave of messages for marginals at query idxs
            # true that extra work is done at idxs where don't care about marginals, but was simplest to just re-use existing BP code
            # TODO in future: would not be too hard, just pre-calculate the F, T, R mxs for each index and only operate with these
            self.run_BP_till_convergence(max_iters=1, verbose=False)
            
            # select marginal dists at query idxs
            log_margs = self.get_univariate_marginals()
            assert log_margs.shape == (n_threads, len(self.G.nodes), len(self.alphabet))
            selected_log_margs = log_margs[range(n_threads), query_idxs, :]
            assert selected_log_margs.shape == (n_threads, len(self.alphabet))
            
            # sample hard vals
            categ = torch.distributions.categorical.Categorical(selected_log_margs.exp())
            sampled_vals = categ.sample().float()
            assert sampled_vals.shape == (n_threads,)

            # insert sampled vals into mx of samples
            new_samples[range(n_threads), query_idxs] = sampled_vals
        
        return new_samples

    def get_conditional_samples(self, query, n, tol_R, timeout):
        orig_num_queries = query.shape[0]
        
        # n is the number of samples to average over, per query
        assert n % self.n_parallel == 0 and n > self.n_parallel
        
        # query is a 2D mx of shape n_samples x n_vars, query vals marked by value -1
        assert query.shape == (orig_num_queries, len(self.G.nodes))
        
        # idea is to sample by freezing observed (conditioned on) variables
        # do burnin, sampled needed number of samples, check mixing condition and repeat if did not mix
        
        # every sample should have the same number of queries for parallel completion
        truth_sum = (query == -1).sum(dim=1)
        num_query_vars = truth_sum[0].item()
        assert (truth_sum / num_query_vars == torch.ones_like(truth_sum)).all()
        
        # each query will be tackled by parallel processes
        query = query.unsqueeze(1).repeat(1,self.n_parallel,1).reshape(orig_num_queries*self.n_parallel, query.shape[1])
        assert query.shape == (self.n_parallel*orig_num_queries, len(self.G.nodes))
        
        # latest sample held in obs
        obs = query.clone()
        
        # samples that could be extracted when returning enter this list
        captures = []

        # start with uar values for query variables
        uar_samples = self.get_uar_samples(shape=obs.shape)
        obs[obs == -1] = uar_samples[obs==-1]
        
        # set bs for the 1 iter of BP done for sampling
        self.bs = orig_num_queries*self.n_parallel
        
        tried = 0
        
        while True:
            for wave in range(self.n_burnin + int(n / self.n_parallel)):
                # orders is a 2D mx, with indices in each row for how to fill each sample
                # order is different per sample, per iteration
                orders = torch.stack((query == -1).nonzero()[:,1].split(num_query_vars))
                assert orders.shape == (self.n_parallel*orig_num_queries, num_query_vars)

                # due to orders ony containing query vars, no need to use mask to update obs
                obs = self.get_one_wave_of_samples(old_samples=obs, orders=orders)
            
                # check that observed vars are indeed frozen
                assert (obs[query != -1] == query[query != -1]).all()
                
                if wave >= self.n_burnin:
                    captures.append(obs.clone())
            
            # want to check mixing separately for each query
            tried += 1
            
            # recall that we had n_parallel rows per actual original query in each sample
            assert len(captures) == int(n / self.n_parallel), print(len(captures), int(n / self.n_parallel))
            for c in captures: assert c.shape == (self.n_parallel*orig_num_queries, len(self.G.nodes))
            captures = [c.reshape(orig_num_queries, self.n_parallel, len(self.G.nodes)) for c in captures]
            
            # to parallelize tol checking, can concatenate all queries horizontally and do one check (each var of each query is a distinct channel)
            # this tensor is then tossed away
            captures_glued = [c.transpose(1,0).reshape(-1, self.n_parallel, orig_num_queries * len(self.G.nodes)).squeeze(0) for c in captures]
            for c in captures_glued: 
                assert c.shape == (self.n_parallel, orig_num_queries * len(self.G.nodes))
            
            if tol_R is not None:
                if self.check_mixed(captures_glued, tol_R):
                    break
                    
            if tried == timeout:
                print('timed out and could not get conditional sampling MC to mix')
                break

            captures = []
            continue
        
        captures = torch.cat(captures, dim=1)
        assert captures.shape == (orig_num_queries, n, len(self.G.nodes))            
        
        return captures


def get_random_edges(n_vars, n_edges_all=None, n_edges_per_v=None):
    variables = list(range(n_vars))    
    assert n_edges_all is None or n_edges_per_v is None
    assert n_edges_all is not None or n_edges_per_v is not None
    
    if n_edges_all is not None:
        edges = random.sample(list(combinations(variables, 2)), n_edges_all)
    else:
        edges = []
        for v in variables:
            edges += random.sample([(v, v_) for v_ in variables if v_ != v], n_edges_per_v)
            
    return edges

def get_empirical_univar_marginals(samples):
    marg = []
    for i in range(samples.shape[1]):
        # force 0 and 1 in each col to prevent counts of 0
        # like laplacian
        col = samples[:, i]
        col = torch.cat([col, torch.FloatTensor([0,1]).to(col.device)])
        
        # count and get dist
        _, counts = torch.unique(col, return_counts=True)
        marg.append(counts.float() / counts.sum())

    marg = torch.stack(marg)
    return marg
