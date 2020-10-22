from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import os
import torch
import numpy as np

from utils import evaluate_standard_test_stub, eval_ERM
from code_p_graph_fast import Pairwise_BP_Computer_Parallel, Pairwise_Binary_Gibbs_Sampler

# BP computer object from our source code used here
def run_BP(graph, dct_edge_to_log_pot, n_bp_steps, qry_mx, device):
    computer = Pairwise_BP_Computer_Parallel(graph, device=device)
    computer.reset_state(obs=qry_mx, dct_edge_to_log_pot=dct_edge_to_log_pot)
    computer.run_BP_till_convergence(tol=1e-5, max_iters=n_bp_steps, verbose=False)
    log_marg = computer.get_univariate_marginals()
    return log_marg

# Runner for all of the ERM BP training and testing
def main_EGM(data_name, train_loader, test_loader, G, publisher, mode_images, n_steps, n_bp_steps, device, lr=1e-2, train_ERM=True, test_frac=False, test_squares=True, test_quads=False, test_corrupt=False, test_every=100, model_save_path=None, model_save_name=None, sample_save_n=None, sample_save_burnin=None, sample_save_path=None):
    bp_log_pots = [Variable(torch.randn(len(G.alphabet), len(G.alphabet)).to(device), requires_grad=True) for e in G.undir_edges]
    bp_optimizers = [torch.optim.Adam([bp_log_pots[i]], lr=lr) for i, e in enumerate(G.undir_edges)]
    
    for step in tqdm(range(n_steps)):
        if step != 0:
            if train_ERM:
                try:
                    batch = next(train_loader_iterator)
                except:
                    train_loader_iterator = iter(train_loader)
                    batch = next(train_loader_iterator)

                batch = batch.to(device)

                inp = batch.clone()
                outp = inp.clone()
                
                hiding_mask = torch.FloatTensor(np.random.choice(a=[-1, 100], size=inp.shape, p=[0.5, 0.5])).to(device)
                inp[hiding_mask == -1] = -1 

                ERM_BP_train, ERM_NLL_train = eval_ERM(producer_method=run_BP, optional_args={'graph': G, 'dct_edge_to_log_pot': {e: bp_log_pots[i] for i, e in enumerate(G.undir_edges)}, 'n_bp_steps': n_bp_steps}, qry_mx=inp, targ_mx=outp, device=device, draw=False)

                [optimizer.zero_grad() for optimizer in bp_optimizers]
                (ERM_NLL_train).backward()
                [optimizer.step() for optimizer in bp_optimizers]
            
            else:
                p_nll = eval_plain_LL(graph=G, dct_edge_to_log_pot=dct_edge_to_log_pot, n_bp_steps=n_bp_steps, data=batch, device=device)
                
                [optimizer.zero_grad() for optimizer in bp_optimizers]
                (p_nll).backward()
                [optimizer.step() for optimizer in bp_optimizers]
                
        # test
        if (step % test_every == 0 or step == n_steps - 1):
            
            with torch.no_grad():
                dct_edge_to_log_pot={e: bp_log_pots[i].data for i, e in enumerate(G.undir_edges)}    
                evaluate_standard_test_stub(step=step, producer_method=run_BP, optional_args={'graph': G, 'dct_edge_to_log_pot': dct_edge_to_log_pot, 'n_bp_steps': n_bp_steps}, test_data_loader=test_loader, publisher=publisher, batch_shorted_at=None, device=device, test_frac=test_frac, test_squares=test_squares, test_quads=test_quads, test_corrupt=test_corrupt, draw=mode_images)
            
            publisher.draw_history()
            
    if model_save_path is not None:
        torch.save({
            'pot_modules': [p for p in bp_log_pots],
            'edges': G.undir_edges,
            'dct_edge_to_log_pot': dct_edge_to_log_pot,
            }, os.path.join(model_save_path, f'{model_save_name}-{step}'))
    
    if sample_save_n is not None and sample_save_path is not None and sample_save_burnin is not None:
        samples_to_threads = 1
        sampler = Pairwise_Binary_Gibbs_Sampler(graph=G, n_parallel=int(sample_save_n/samples_to_threads), n_burnin=sample_save_burnin, device=device, dct_edge_to_log_pot={e: bp_log_pots[i].data for i, e in enumerate(G.undir_edges)} )
        samples = sampler.get_samples(n=sample_save_n, tol_R=1e10, timeout=1)
        torch.save(samples.detach().cpu(), os.path.join(sample_save_path, f'samples-{data_name}-{sample_save_n}-{sample_save_burnin}'))
        print(f'saved {sample_save_n} sampled pts to {sample_save_path}, after burnin {sample_save_burnin}, samples spread over {samples_to_threads} threads')
        
    return publisher
