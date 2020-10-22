import torch
from torch import nn
from torch.autograd import Variable

import os
from tqdm import tqdm as tqdm

from utils import evaluate_standard_test_stub, plot_some_imgs


### Modules

# x to z
class Encoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        self.model = nn.Sequential(
            nn.Linear(in_features=d_in, out_features=d_in*4),
            nn.BatchNorm1d(d_in*4),
            nn.LeakyReLU(negative_slope=0.1),
            
            # * 2 to get mu and sigma
            nn.Linear(in_features=d_in*4, out_features=d_out*2))

    def forward(self, x):
        bs = x.shape[0]
        assert x.shape == (bs, self.d_in)

        res = self.model(x)
        mu = res[:, :self.d_out]
        sigma = res[:, self.d_out:]
            
        # re-parametrization trick for sampling from gaussians
        return mu + Variable(torch.randn([bs, self.d_out])).to(sigma.device) * sigma

# z to x
class Decoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(Decoder, self).__init__()
            
        self.d_in = d_in
        self.d_out = d_out
        c = 2
        
        self.model = nn.Sequential(
            nn.Linear(in_features=d_in, out_features=d_in*4*c),
            nn.BatchNorm1d(d_in*4*c),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(in_features=d_in*4*c, out_features=d_in*4*c),
            nn.BatchNorm1d(d_in*4*c),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(in_features=d_in*4*c, out_features=d_in*4*c),
            nn.BatchNorm1d(d_in*4*c),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(in_features=d_in*4*c, out_features=d_out),
            nn.Sigmoid())

    def forward(self, z):
        bs = z.shape[0]
        assert z.shape == (bs, self.d_in)
        probs = self.model(z).unsqueeze(2)
        converse = 1 - probs
        return torch.cat([probs, converse], dim=2)


class Discriminator(nn.Module):
    def __init__(self, d_x, d_z):
        super(Discriminator, self).__init__()
        self.d_x = d_x
        self.d_z = d_z

        self.D_x = nn.Sequential(
            nn.Linear(in_features=d_x, out_features=d_x*4),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.1),)

        self.D_z = nn.Sequential(
            nn.Linear(in_features=d_z, out_features=d_z*4),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.1),)

        self.D_xz = nn.Sequential(
            nn.Linear(in_features=d_x*4 + d_z*4, out_features=d_x*4 + d_z*4),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(in_features=d_x*4 + d_z*4, out_features=1),
            nn.Dropout(0.2))

    def forward(self, x, z):
        D_x = self.D_x(x)
        D_z = self.D_z(z)
        D_xz = torch.cat((D_x, D_z), 1)
        return self.D_xz(D_xz)

    
### Model

class GibbsNet_WGAN_GP_with_BP():
    def __init__(self, n_vars, z_dimension, device, lr):
        self.n_vars = n_vars
        self.device = device
        
        self.encoder = Encoder(d_in=n_vars*2, d_out=z_dimension).to(device)
        self.decoder = Decoder(d_in=z_dimension, d_out=n_vars).to(device)
        self.discriminator = Discriminator(d_x=n_vars*2, d_z=z_dimension).to(device)
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0, 0.9))
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(0, 0.9))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0, 0.9))

        # gaussian initialization
        self.encoder.apply(normal_weight_init)
        self.decoder.apply(normal_weight_init)
        self.discriminator.apply(normal_weight_init)
    
    # test args passed in rather than being in opt, as may want to train one way and test several ways
    def produce(self, conditions, query_sample_size, n_iters, condition_before_graph=True):
        n_queries = conditions.shape[0]
        gaussian_z = torch.randn((n_queries*query_sample_size, self.encoder.d_out)).to(self.device)
        conditions = conditions.unsqueeze(1).repeat(1, query_sample_size, 1).reshape(n_queries*query_sample_size, self.n_vars)
        conditions_with_soft_samples = hard_samples_to_soft_samples(conditions)
        
        for i in range(n_iters):
            # z -> x
            sampled_x = self.decoder(z=gaussian_z)
            sampled_x[conditions != -1] = conditions_with_soft_samples[conditions != -1]                
            # x -> z, ignore last iter
            assert sampled_x.shape == (n_queries*query_sample_size, self.n_vars, 2)
            gaussian_z = self.encoder(sampled_x.reshape(n_queries*query_sample_size, self.n_vars*2))
        
        return sampled_x.reshape(n_queries, query_sample_size, self.n_vars, 2)
        
    def save(self, path, name, step):
        torch.save({
            'step': step,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            }, os.path.join(path, f'{name}-{str(step)}'))
        
    def load(path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        return
        
def normal_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

        
### Helpers

# assume a binary alphabet
def hard_samples_to_soft_samples(x):
    bs, n_vars = x.shape
    soft = torch.zeros(bs, n_vars, 2)
    soft[x == 0] = torch.FloatTensor([1,0])    
    soft[x == 1] = torch.FloatTensor([0,1])
    return soft.to(x.device)

# soft = hard_samples_to_soft_samples(torch.FloatTensor([[1,0,0],[0,1,1],[1,0,1]]))


### Runner

from torch.autograd import grad as get_torch_grad

def main_GibbsNet(data_name, train_loader, test_loader, G, publisher, mode_images, z_dimension, lamb, sampling_count, ratio_D_to_G, lr, n_steps, device, test_frac=True, test_squares=True, test_quads=True, test_corrupt=True, test_every=None, model_save_path=None, model_save_name=None):
        
    # return 2D samples mx (n_queries, n_vars) and a 3D marg mx (n_queries, n_vars, alphabet size)
    def conditional_sampler(qry_mx, query_sample_size, n_burnin, device=None):
        # marginals where each query has n_per_query corresponding result rows
        marg = model.produce(conditions=qry_mx, query_sample_size=query_sample_size, n_iters=n_burnin)
        assert marg.shape == (qry_mx.shape[0], query_sample_size, n_vars, len(alphabet))
        
        # mean of marginals
        marg_mean = marg.mean(dim=1)
        assert marg_mean.shape == (qry_mx.shape[0], n_vars, len(alphabet))
        return marg_mean.log()
    
    def get_gradient_penalty(model, real_pair, fake_pair):
        (real_x, sampled_z) = real_pair
        (sampled_x, gaussian_z) = fake_pair
        assert real_x.shape[0] == sampled_z.shape[0] == sampled_x.shape[0] == gaussian_z.shape[0]
        
        eps = torch.rand(real_x.shape[0]).unsqueeze(1).to(real_x.device)
        hat_x = eps * real_x + (1 - eps) * sampled_x
        hat_z = eps * sampled_z + (1 - eps) * gaussian_z
        d_hat = model.discriminator(hat_x, hat_z)

        # Calculate gradient of discriminator(hat_x) wrt. hat_x
        # outputs a list, here of only one thing as we have only one input = hat_x
        gradients = get_torch_grad(outputs=d_hat, inputs=[hat_x, hat_z], grad_outputs=torch.ones(d_hat.size()).to(d_hat.device), create_graph=True, retain_graph=True)

        assert len(gradients) == 2
        assert gradients[0].shape == hat_x.shape
        assert gradients[1].shape == hat_z.shape
        
        gradients = torch.cat(gradients, dim=1)
        
        # want norms per row
        # derivatives close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
    
    alphabet = G.alphabet
    n_vars = len(G.nodes)
    
    if z_dimension is None:
        if len(G.nodes) < 500: z_dimension = 64
        else: z_dimension = 128
    
    model = GibbsNet_WGAN_GP_with_BP(n_vars=n_vars, z_dimension=z_dimension, device=device, lr=lr)
    
    for step in tqdm(range(n_steps)):
        
        try: real_x = next(dl).to(device)
        except: 
            dl = iter(train_loader)
            real_x = next(dl).to(device)
        
        real_x = hard_samples_to_soft_samples(real_x)
        bs = real_x.shape[0]
        assert real_x.shape == (bs, n_vars, 2)
        
        real_x = real_x.reshape(bs, n_vars*2)
        
        ## user-provided x to its produced z
        sampled_z = model.encoder(real_x)
        
        ## gaussian noise z to artificial x: z_0 -> x_hat_0 -> z_1 -> ... -> x_hat_n -> z_n
        # unclamped chain: no gradient
        
        with torch.no_grad():
            gaussian_z = torch.randn((bs, z_dimension)).to(device)

            for i in range(sampling_count):
                sampled_x = model.decoder(gaussian_z)
                assert sampled_x.shape == (bs, n_vars, 2)
                gaussian_z = model.encoder(sampled_x.reshape(bs, n_vars*2))
        
        # clamped chain
        # last z -> x_fake step, which carries gradient
        sampled_x = model.decoder(gaussian_z)
        assert sampled_x.shape == (bs, n_vars, 2)
        sampled_x = sampled_x.reshape(bs, n_vars*2)
        
        ## Note that if only sampling, can skip the following block
        # discriminate
        d_real_x = model.discriminator(real_x, sampled_z)
        d_fake_x = model.discriminator(sampled_x, gaussian_z)
        
        if lamb != 0:
            penalty = lamb * get_gradient_penalty(model, real_pair=(real_x, sampled_z), fake_pair=(sampled_x, gaussian_z))
        else:
            penalty = 0
        
        # losses and back propagation
        
        # update discriminator: must guess (real_x, sampled_z) as real and (sampled_x, gaussian_z) as fake
        model.discriminator_optimizer.zero_grad()
        
        D_loss = -torch.mean(d_real_x) + torch.mean(d_fake_x) + penalty
        
        if step % ratio_D_to_G == 0:
            D_loss.backward(retain_graph=True)
        else:
            D_loss.backward(retain_graph=False)
        
        model.discriminator_optimizer.step()       
        
        if step % ratio_D_to_G == 0:
            # update generator: want to fool discriminator so want to 
            # make (real_x, sampled_z) seem fake and (sampled_x, gaussian_z) seem real        
            model.encoder_optimizer.zero_grad()
            model.decoder_optimizer.zero_grad()

            G_loss = torch.mean(d_real_x) - torch.mean(d_fake_x)
            G_loss.backward()

            model.encoder_optimizer.step()
            model.decoder_optimizer.step()
        
        if (step % test_every == 0 or step == n_steps-1):
            
            with torch.no_grad():
                    
                evaluate_standard_test_stub(step=step, producer_method=conditional_sampler, optional_args={'query_sample_size':100, 'n_burnin':sampling_count}, batch_shorted_at=None, test_data_loader=test_loader, publisher=publisher, device=device, test_frac=test_frac, test_squares=test_squares, test_quads=test_quads, test_corrupt=test_corrupt, draw=mode_images)
                publisher.draw_history()
                
                if model_save_path is not None:
                    model.save(path=model_save_path, name=model_save_name, step=step)
    return
