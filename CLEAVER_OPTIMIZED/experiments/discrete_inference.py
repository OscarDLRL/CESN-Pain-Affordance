"""SAFE/PAIN is the only inferred variable. XY and static latent are known inputs."""
import torch
import numpy as np
import h5py
from CLEAVER_OPTIMIZED import config as cfg

@torch.no_grad()
def perceived_xy(d, perception, lo, hi, source=cfg.RGB):
    with h5py.File(source, 'r') as f:
        p = np.stack([perception.goal_xy(f[k]['rgb'][cfg.REFERENCE_FRAME])[0] for k in d['keys']])
    p = torch.as_tensor(p, dtype=torch.float32, device=lo.device)
    return 2*(p-lo)/(hi-lo)-1, p.cpu().numpy()

class Incremental:
    def __init__(self, model, d, xy, k):
        self.m, self.k = model, k
        self.B = len(d['keys'])
        B = self.B
        labs = torch.arange(2, device=xy.device).repeat_interleave(B)
        c = model.context(labs, d['static_n'].repeat(2,1), xy.repeat(2,1))
        self.base = c @ model.Win + model.bias
        self.const, self.qr = model._readout_parts(c)
        self.r = c.new_zeros(2*B, model.n)
        self.prev = c.new_zeros(2*B, model.Wf.shape[0])
        self.sq = c.new_zeros(2*B)
        self.t = self.count = 0
    @torch.no_grad()
    def step(self, z=None):
        self.r = self.m._step(self.base, self.r, self.prev)
        y = self.const + self.r @ self.qr
        if self.t % self.k == 0:
            if z is None: raise ValueError('Missing scheduled observation')
            z = z.repeat(2,1)
            self.sq += (y-z).square().mean(1)
            self.count += 1
            self.prev = z
        else:
            self.prev = y
        self.t += 1
        return (self.sq / self.count).view(2,self.B).T

@torch.no_grad()
def fixed_losses(model,d,xy,n,k):
    B=len(d['keys'])
    labs=torch.arange(2,device=xy.device).repeat_interleave(B)
    c=model.context(labs,d['static_n'].repeat(2,1),xy.repeat(2,1))
    obs=d['yn'][:,:n].repeat(2,1,1)
    p=model.predict(c,n,obs,k)
    return (p[:,::k]-obs[:,::k]).square().mean((1,2)).view(2,B).T

@torch.no_grad()
def history(model,d,xy,k):
    state=Incremental(model,d,xy,k)
    return torch.stack([state.step(d['yn'][:,t] if t%k==0 else None) for t in range(d['yn'].shape[1])],1)
