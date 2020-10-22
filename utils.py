### Data loading and processing

from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np

class Custom_Dataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def load_data(name, train_bs, test_bs, custom_train_data_path=None, custom_test_data_path=None, cap_train=None, cap_test=None):
    if name == 'MNIST':
        train_data, train_labels = torch.load('./datasets/MNIST/training.pt')
        test_data, test_labels = torch.load('./datasets/MNIST/testing.pt')

        train_data = train_data.reshape(-1, 784).float()
        test_data = test_data.reshape(-1, 784).float()

        train_data[train_data < 30] = 0
        train_data[train_data >= 30] = 1

        test_data[test_data < 30] = 0
        test_data[test_data >= 30] = 1
        
        train_data = train_data.float()
        test_data = test_data.float()
    
    elif name == 'CALTECH':
        mat = scipy.io.loadmat('./datasets/Caltech/caltech101_silhouettes_28_split1.mat')
        
        train_data = torch.FloatTensor(mat['train_data'])
        train_labels = torch.LongTensor(mat['train_labels']).squeeze()

        test_data = torch.FloatTensor(mat['test_data'])
        test_labels = torch.FloatTensor(mat['test_labels']).squeeze()

    elif name is not None:
        train_data = torch.FloatTensor(np.loadtxt(f'./datasets/{name}-train.csv', delimiter=','))
        test_data = torch.FloatTensor(np.loadtxt(f'./datasets/{name}-test.csv', delimiter=','))
        
    # can provide a name as above, then overwrite train or test alone by some custom
    # or just give None for a name and load customs here
    if custom_train_data_path is not None:
        train_data = torch.FloatTensor(torch.load(custom_train_data_path))
        print(f'loaded custom train data from {custom_train_data_path}')
    if custom_test_data_path is not None:
        test_data = torch.FloatTensor(torch.load(custom_test_data_path))
        print(f'loaded custom test data from {custom_test_data_path}')
        
    n_vars = train_data.shape[1]
    variables = list(range(n_vars))
    
    if cap_train is not None: train_data = train_data[:cap_train] 
    if cap_test is not None: test_data = test_data[:cap_test] 
        
    train_loader = DataLoader(dataset=Custom_Dataset(train_data), batch_size=train_bs, pin_memory=True, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=Custom_Dataset(test_data), batch_size=test_bs, pin_memory=True, shuffle=False, num_workers=0, drop_last=True)
    
    return train_loader, test_loader, variables

def get_grid_edges(width):
    import networkx as nx
    G = nx.grid_2d_graph(width, width, periodic=False, create_using=None)
    G.remove_edges_from([(i,i) for i in range(width*width)])
    return [(e1[0]*width+e1[1], e2[0]*width+e2[1])  for (e1,e2) in G.edges]


### Plotting and tracking results

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import math as m
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_some_imgs(imgs, n_cols=5, soft=False):
    n = imgs.shape[0]
    n_rows = m.ceil(n / n_cols)
    
    w_img = int(m.sqrt(imgs.shape[1]))
    
    fig = plt.figure(figsize=(20,20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                 axes_pad=0.1)  # pad between axes in inch.
    
    for ax, im in zip(grid, imgs.reshape(n,w_img,w_img)):
        if soft or ((im > 0).float() * (im < 1).float()).sum() != 0: 
            cmap = 'gray'
            norm=None
        else:
            # make a color map of fixed colors with 0: black, 1: white, -1: red
            cmap = colors.ListedColormap(['red', 'black', 'white'])
            bounds=[-1.5,-0.5,0.5,1.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
        
        ax.imshow(im, cmap=cmap, norm=norm, vmin=0, vmax=1)
        
        # hide ticks and axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
    plt.show()
    return

class Publisher():
    def __init__(self):
        self.history = {}
    
    def record(self, title, infos):
        if title not in self.history: self.history[title] = {}
            
        for info in infos:
            if info['name'] not in self.history[title]: self.history[title][info['name']] = {'data': [], 'best_fn': info["best_fn"]}
            self.history[title][info['name']]['data'].append((info['epoch'], info['val']))
        return
    
    def draw_history(self, specific_titles=None, background_color='white', save_name=None):
        if specific_titles is None: specific_titles = list(self.history.keys())
            
        import math as m
        num_plots = len(specific_titles)
        nc = 4
        nr = m.ceil(num_plots / nc)
        
        f = plt.figure(figsize=(25,5))
    
        for i, title in enumerate(specific_titles):                
            ax = f.add_subplot(nr, nc, i+1)
            ax.set_title(title)
            
            for subtitle in self.history[title]:
                ax.plot([l[0] for l in self.history[title][subtitle]["data"]], [l[1] for l in self.history[title][subtitle]["data"]], label=subtitle)
                
                print(f'best {subtitle}: {self.history[title][subtitle]["best_fn"](self.history[title][subtitle]["data"], key=lambda x: x[1])}')
                
            ax.set_xlabel("epochs")
            ax.legend()
        
        f.set_facecolor(background_color)
        if save_name: plt.savefig(save_name)
        plt.show()
        
        return
        

### Running tests

def eval_ERM(producer_method, qry_mx, targ_mx, device, optional_args={}, draw=True):
    # queries fed in as observations, with -1 at query idxs
    log_marg = producer_method(qry_mx=qry_mx, device=device, **optional_args)
    
    # plot if images
    # qry then target then marg then hard sample so can easily cut off qry and hard sample if not needed
    if draw: 
        imgs = torch.cat([targ_mx[:5].cpu(), qry_mx[:5].cpu(), log_marg[:5, :, 1].exp().detach().cpu(), torch.argmax(log_marg, dim=2)[:5].cpu().float()], dim=0)
        plot_some_imgs(imgs, n_cols=5)
    
    log_marg = log_marg.reshape(-1, log_marg.shape[2])
    
    # get predictions from marginals at query idxs
    msk = qry_mx.reshape(-1) == -1
    log_marg_sel = log_marg[msk]
    targ_sel = targ_mx.reshape(-1)[msk]

    # accuracy
    pred_sel = torch.argmax(log_marg_sel, dim=1).float()    
    accuracy = (targ_sel==pred_sel).sum().float() * (100 / msk.sum().float())
    nll = torch.nn.NLLLoss(reduction='mean')(log_marg_sel, targ_sel.long())
    return accuracy, nll


def remove_squares(imgs, w=4):
    bs = imgs.shape[0]
    width = int(m.sqrt(imgs.shape[1]))
    imgs = imgs.clone().reshape(bs, width, width)
    upper_left_row = m.floor((width-w)/2)
    upper_left_col = m.floor((width-w)/2)
    imgs[:, upper_left_row:upper_left_row+w, upper_left_col:upper_left_col+w] = -1
    return imgs.reshape(bs, width*width)

def remove_quadrants(imgs, which):
    imgs = imgs.clone()
    assert int(m.sqrt(imgs.shape[1])) == m.sqrt(imgs.shape[1])
    w = int(m.sqrt(imgs.shape[1]))
    imgs = imgs.reshape(-1, w, w)
    
    cut = int(w/2)
    if which == 0: imgs[:, :cut, cut:] = -1 
    elif which == 1: imgs[:, cut:, cut:] = -1 
    elif which == 2: imgs[:, cut:, :cut] = -1 
    elif which == 3: imgs[:, :cut, :cut] = -1 
    else: raise NotImplementedError
        
    return imgs.reshape(-1, w*w)

def remove_fraction(imgs, f):
    imgs = imgs.clone()
    hiding_mask = torch.FloatTensor(np.random.choice(a=[-1, 100], size=imgs.shape, p=[f, 1-f])).to(imgs.device)
    imgs[hiding_mask == -1] = -1 
    return imgs

def remove_fraction_and_corrupt(imgs, p, f):
    imgs = imgs.clone()
    # 1 if corrupted, 0 if not
    corrupt_mask = torch.FloatTensor(np.random.choice(a=[1, 0], size=imgs.shape, p=[p, 1-p])).to(imgs.device)
    imgs += corrupt_mask
    imgs = imgs % 2

    imgs = remove_fraction(imgs, f)
    return imgs
    
# combines the above tests into a test suite, kept same across models
def evaluate_standard_test_stub(step, producer_method, optional_args, test_data_loader, publisher, device, batch_shorted_at=None, test_frac=True, test_squares=True, test_quads=True, test_corrupt=True, draw=False):
    # standard test stub for paper
    fracs = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    ERM_fracs = [0 for frac in fracs]
    ERM_NLL_fracs = [0 for frac in fracs]
    
    Ws = list(range(4,10))
    ERM_Ws = [0 for w in Ws]
    ERM_NLL_Ws = [0 for w in Ws]
    
    quads = [0,1,2,3]
    ERM_quads = [0 for q in quads]
    ERM_NLL_quads = [0 for q in quads]
    
    Ps = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    ERM_Ps = [0 for p in Ps]
    ERM_NLL_Ps = [0 for p in Ps]
    
    n_batches = 0

    for batch in iter(test_data_loader):
        batch = batch.to(device)
        
        if test_frac:
            for i, frac in enumerate(fracs):
                inp = remove_fraction(imgs=batch.clone(), f=fracs[i])
                outp = batch.clone()
                # 70% all hidden
                acc, e_nll = eval_ERM(producer_method=producer_method, qry_mx=inp, targ_mx=outp, draw=draw and n_batches==0, device=device, optional_args=optional_args)
                ERM_fracs[i] += acc
                ERM_NLL_fracs[i] += e_nll
        
        if test_squares:
            for i, w in enumerate(Ws):
                q_squares = remove_squares(batch.clone(), w=w)
                acc, e_nll = eval_ERM(producer_method=producer_method, qry_mx=q_squares, targ_mx=batch, draw=draw and n_batches==0, device=device, optional_args=optional_args)
                ERM_Ws[i] += acc
                ERM_NLL_Ws[i] += e_nll
                
        if test_quads:
            for i, q in enumerate(quads):
                q_quad = remove_quadrants(batch.clone(), which=q)
                acc, e_nll = eval_ERM(producer_method=producer_method, qry_mx=q_quad, targ_mx=batch, draw=draw and n_batches==0, device=device, optional_args=optional_args)
                ERM_quads[i] += acc
                ERM_NLL_quads[i] += e_nll
        
        if test_corrupt:
            for i, p in enumerate(Ps):
                q_corrupt_hidden = remove_fraction_and_corrupt(batch.clone(), p, f=0.5)
                acc, e_nll = eval_ERM(producer_method=producer_method, qry_mx=q_corrupt_hidden, targ_mx=batch, draw=draw and n_batches==0, device=device, optional_args=optional_args)
                ERM_Ps[i] += acc
                ERM_NLL_Ps[i] += e_nll
        
        n_batches += 1
        if batch_shorted_at is not None and n_batches == batch_shorted_at: break
    
    if test_frac:
        for i, frac in enumerate(fracs):
            publisher.record(title='ERM fracs acc', infos=[{'name': f'ERM fracs acc {frac}', 'best_fn': max, 'epoch': step, 'val': ERM_fracs[i] / n_batches}])
            publisher.record(title='ERM fracs NLL', infos=[{'name': f'ERM fracs NLL {frac}', 'best_fn': min, 'epoch': step, 'val': ERM_NLL_fracs[i] / n_batches}])
    
    if test_squares:
        for i, w in enumerate(Ws):
            publisher.record(title='ERM squares acc', infos=[{'name': f'ERM squares acc {w}', 'best_fn': max, 'epoch': step, 'val': ERM_Ws[i] / n_batches}])
            publisher.record(title='ERM squares NLL', infos=[{'name': f'ERM squares NLL {w}', 'best_fn': min, 'epoch': step, 'val': ERM_NLL_Ws[i] / n_batches}])
            
    if test_quads:
        for i, q in enumerate(quads):
            publisher.record(title='ERM quads acc', infos=[{'name': f'ERM quads acc {q}', 'best_fn': max, 'epoch': step, 'val': ERM_quads[i] / n_batches}])
            publisher.record(title='ERM quads NLL', infos=[{'name': f'ERM quads NLL {q}', 'best_fn': min, 'epoch': step, 'val': ERM_NLL_quads[i] / n_batches}])
    
    if test_corrupt:
        for i, p in enumerate(Ps):
            publisher.record(title='ERM corrupt acc', infos=[{'name': f'ERM corrupt acc {p}', 'best_fn': max, 'epoch': step, 'val': ERM_Ps[i] / n_batches}])
            publisher.record(title='ERM corrupt NLL', infos=[{'name': f'ERM corrupt NLL {p}', 'best_fn': min, 'epoch': step, 'val': ERM_NLL_Ps[i] / n_batches}])
    
    return