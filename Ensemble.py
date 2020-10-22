import torch
from torch import nn 
from tqdm import tqdm_notebook as tqdm
from torch.autograd import grad as get_torch_grad
import os

from utils import evaluate_standard_test_stub, plot_some_imgs
from code_p_graph_fast import Pairwise_BP_Computer_Parallel

### modules for adversarial pipeline

# z to psi
class Decoder_z_to_psi(nn.Module):
    def __init__(self, d_in, d_out):
        super(Decoder_z_to_psi, self).__init__()
            
        self.d_in = d_in
        self.d_out = d_out

        self.model = nn.Sequential(
            nn.Linear(in_features=d_in, out_features=d_in*2),
            nn.BatchNorm1d(d_in*2),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(in_features=d_in*2, out_features=d_out))

    def forward(self, z):
        bs = z.shape[0]
        assert z.shape == (bs, self.d_in)
        return self.model(z)
    
# psi to x
class Decoder_psi_to_x(Pairwise_BP_Computer_Parallel):
    def __init__(self, graph, n_bp_steps, device):
        super(Decoder_psi_to_x, self).__init__(graph, device)
        self.n_bp_steps = n_bp_steps
        self.alphabet = [0,1]
        self.device = device
        
        return
    
    def format_z_from_decoder_to_BP_ready_psi(self, z):
        bs = z.shape[0]
        assert z.shape == (bs, len(self.G.undir_edges)*len(self.alphabet)*len(self.alphabet))
        
        # want to repeat values so that get log pots for every edge
        # meaning edges between same nodes but in opposite directions must get transposes of each other as log pots
        indices = []
        for i in range(len(self.G.undir_edges)):
            indices += [(i*4)+0, (i*4)+1, (i*4)+2, (i*4)+3]
            indices += [(i*4)+0, (i*4)+2, (i*4)+1, (i*4)+3]
        z = z.index_select(dim=1, index=torch.LongTensor(indices).to(self.device))
        psi = z.reshape(bs, len(self.G.edges), len(self.alphabet), len(self.alphabet)).permute(0,2,3,1)
        assert psi.shape == (bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
        return psi
        
    def decode_through_BP(self, z, conditions=None):
        # z has log pots for every UNDIR edge
        # set bs for the parent class methods
        self.bs = z.shape[0]
        assert z.shape == (self.bs, len(self.G.undir_edges)*len(self.alphabet)*len(self.alphabet))
        
        # NOTE: right now we are predicting the LOG pots directly, so we do not explicitly take a log here
        self.psi = self.format_z_from_decoder_to_BP_ready_psi(z)
        assert self.psi.shape == (self.bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
        
        # initial msgs mx
        self.M = torch.ones(self.bs, len(self.alphabet), len(self.G.edges)).to(self.device).log()
        assert self.M.shape == (self.bs, len(self.alphabet), len(self.G.edges))
    
        # node log pots uniform if nothing observed
        # if conditioning, note that mask is ready to be fed as obs to psi constructor 
        # (-1 = to be queries, other vals = observed)
        if conditions is None: conditions = torch.ones(self.bs, len(self.G.nodes)) * -1            
        self.phi = self.get_node_log_pots(obs=conditions, dct_var_to_log_pot={}, uniform_alphabet=True)
        assert self.phi.shape == (self.bs, len(self.alphabet), len(self.G.nodes))
        
        # BP for marginals from which we get x
        self.run_BP_till_convergence(tol=1e-5, max_iters=self.n_bp_steps, verbose=False)
        log_marg = self.get_univariate_marginals()
        assert log_marg.shape == (self.bs, len(self.G.nodes), len(self.alphabet))

        sampled_vals = torch.softmax(log_marg, dim=2)
            
        # we also return the log marginals, because during testing, (e.g. say we are answering queries)
        # we might care say about the most likely outcome at each node
        # since we would no longer care about differentiability etc
        return sampled_vals

class Discriminator(nn.Module):
    def __init__(self, d_x):
        super(Discriminator, self).__init__()
        self.d_x = d_x

        self.D_x = nn.Sequential(
            nn.Linear(in_features=d_x, out_features=d_x*2),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(in_features=d_x*2, out_features=d_x*2),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(in_features=d_x*2, out_features=1),
            nn.Dropout(0.2))

    def forward(self, x):
        return self.D_x(x)

### tiny helpers

def convert_uni_to_bi_channels(x):
    bs, n_vars = x.shape
    soft = torch.stack([1-x, x], dim=2)
    return soft.to(x.device)

def normal_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


### main model class
class AGM():
    def __init__(self, G, z_dimension, lr, n_bp_steps, device):
        self.G = G
        self.z_dimension = z_dimension
        self.alphabet = [0,1]
        self.device = device
        self.decoder_z_to_psi = Decoder_z_to_psi(d_in=z_dimension, d_out=len(G.undir_edges)*len(self.alphabet)*len(self.alphabet)).to(device)
        self.decoder_psi_to_x = Decoder_psi_to_x(graph=G, n_bp_steps=n_bp_steps, device=device)
        self.discriminator = Discriminator(d_x=len(self.G.nodes)*2).to(device)
        
        self.decoder_z_to_psi_optimizer = torch.optim.Adam(self.decoder_z_to_psi.parameters(), lr=lr, betas=(0, 0.9))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0, 0.9))

        # gaussian initialization
        self.decoder_z_to_psi.apply(normal_weight_init)
        self.discriminator.apply(normal_weight_init)

    def extract_psis_no_grounding(self, bs):
        gaussian_z = torch.randn((bs, self.z_dimension)).to(self.device)
        sampled_psi = self.decoder_z_to_psi(gaussian_z)
        assert sampled_psi.shape == (bs, len(self.alphabet)*len(self.alphabet)*len(self.G.undir_edges))
        
        sampled_psi = self.decoder_psi_to_x.format_z_from_decoder_to_BP_ready_psi(z=sampled_psi)
        assert sampled_psi.shape == (bs, len(self.alphabet), len(self.alphabet), len(self.G.edges))
        return sampled_psi
    
    def save(self, path, name):
        torch.save({
            'decoder_z_to_psi': self.decoder_z_to_psi.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            }, os.path.join(path, f'{name}'))
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.decoder_z_to_psi.load_state_dict(checkpoint['decoder_z_to_psi'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        return

### run BP given ensemble of params, for testing
def run_ensemble_BP(graph, psis, n_bp_steps, qry_mx, device):
    assert len(psis.shape) == 4
    ensemble_size = psis.shape[0]
    bs = qry_mx.shape[0]
    
    psis = psis.repeat(bs, 1, 1, 1)

    assert len(qry_mx.shape) == 2
    qry_mx = qry_mx.unsqueeze(1).repeat(1, ensemble_size, 1).reshape(-1, qry_mx.shape[-1])
    
    computer = Pairwise_BP_Computer_Parallel(graph, device=device)
    computer.reset_state(obs=qry_mx, psi=psis)
    computer.run_BP_till_convergence(tol=1e-5, max_iters=n_bp_steps, verbose=False)
    log_marg = computer.get_univariate_marginals()
    
    assert log_marg.shape[0] == bs*ensemble_size
    
    target_size = tuple([bs, ensemble_size] + list(log_marg.shape[1:]))
    log_marg = log_marg.reshape(target_size)
    
    # pooling method defined in our paper
    log_marg = log_marg.mean(dim=1) 
    
    # since from here, all we do is take argmax, same if we do argmax of normalized version
    log_marg = log_marg - log_marg.logsumexp(dim=-1, keepdim=True)
    
    return log_marg


### run adversarial training and testing pipeline
def main_AGM(data_name, train_loader, test_loader, G, publisher, mode_images, z_dimension, M, lr, n_bp_steps, lamb, ratio_D_to_G, n_steps, device, test_frac=True, test_squares=True, test_quads=True, test_corrupt=True, test_every=5000, model_save_path=None, model_save_name=None, sample_save_n=None, sample_save_path=None):
    data_size = train_loader.dataset[0:1].shape[1]
    
    if z_dimension is None:    
        if data_size < 500: z_dimension = 64
        else: z_dimension = 128
    
    model = AGM(G=G, z_dimension=z_dimension, lr=lr, n_bp_steps=n_bp_steps, device=device)

    
    def get_gradient_penalty(model, real_x, fake_x):        
        eps = torch.rand(real_x.shape[0]).unsqueeze(1).to(real_x.device)
        hat_x = eps * real_x + (1 - eps) * sampled_x
        d_hat = model.discriminator(hat_x)

        # Calculate gradient of discriminator(hat_x) wrt. hat_x
        # outputs a list, here of only one thing as we have only one input = hat_x
        gradients = get_torch_grad(outputs=d_hat, inputs=[hat_x], grad_outputs=torch.ones(d_hat.size()).to(d_hat.device), create_graph=True, retain_graph=True)

        # want norms per row
        # derivatives close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients[0] ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
    
    dl = iter(train_loader)
    
    for step in tqdm(range(n_steps)):
        model.decoder_z_to_psi.train()
        model.discriminator.train()

        try: real_x = next(dl).to(device)
        except: 
            dl = iter(train_loader)
            real_x = next(dl).to(device)
        
        bs = real_x.shape[0]
        real_x = convert_uni_to_bi_channels(real_x)
        assert real_x.shape == (bs, data_size, 2), real_x.shape

        real_x = real_x.reshape(bs, data_size*2)

        gaussian_z = torch.randn((bs, z_dimension)).to(device)

        # generate
        sampled_psi = model.decoder_z_to_psi(gaussian_z).reshape(bs, -1)
        sampled_x = model.decoder_psi_to_x.decode_through_BP(sampled_psi)
        assert sampled_x.shape == (bs, data_size, 2)
        sampled_x = sampled_x.reshape(bs, data_size*2)

        # discriminate
        d_real_x = model.discriminator(real_x)
        d_fake_x = model.discriminator(sampled_x)
        penalty = get_gradient_penalty(model, real_x, sampled_x)


        # update discriminator
        model.discriminator_optimizer.zero_grad()

        D_loss = -torch.mean(d_real_x) + torch.mean(d_fake_x) + lamb * penalty

        if step % ratio_D_to_G == 0:
            D_loss.backward(retain_graph=True)
        else:
            D_loss.backward(retain_graph=False)

        model.discriminator_optimizer.step()       

        if step % ratio_D_to_G == 0:
            # update generator
            model.decoder_z_to_psi_optimizer.zero_grad()

            G_loss = torch.mean(d_real_x) - torch.mean(d_fake_x)
            G_loss.backward()

            model.decoder_z_to_psi_optimizer.step()

        if (step % test_every == 0 or step == n_steps-1):

            with torch.no_grad():
                model.decoder_z_to_psi.eval()
                model.discriminator.eval()
                psis = model.extract_psis_no_grounding(bs=M)
                    
                evaluate_standard_test_stub(step=step, producer_method=run_ensemble_BP, optional_args={'graph': G, 'psis': psis, 'n_bp_steps': n_bp_steps}, test_data_loader=test_loader, publisher=publisher, batch_shorted_at=None, device=device, test_frac=test_frac, test_squares=test_squares, test_quads=test_quads, test_corrupt=test_corrupt, draw=mode_images)
                publisher.draw_history()
                
    if model_save_path is not None and model_save_name is not None:
        model.save(model_save_path, f'{model_save_name}-{step}')
        print(f'saved model {model_save_name} to {model_save_path}')
    
    if sample_save_n is not None and sample_save_path is not None:
        z = torch.randn(sample_save_n, z_dimension).to(device)
        sampled_psi = model.decoder_z_to_psi(z)
        sampled_x = model.decoder_psi_to_x.decode_through_BP(sampled_psi)
        data = torch.argmax(sampled_x, dim=2).float()
        torch.save(data.detach().cpu(), os.path.join(sample_save_path, f'samples-{data_name}-{sample_save_n}'))
        print(f'saved {sample_save_n} sampled pts to {sample_save_path}')
    
    return publisher  
