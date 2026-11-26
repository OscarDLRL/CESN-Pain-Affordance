"""V6-AL encoder-sharing ablation. Run from repository root; validation only."""
import argparse, hashlib, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from CLEAVER_OPTIMIZED.experiments import train_v6_compact128 as v
from CLEAVER_OPTIMIZED.experiments.run_v6_residual_ablation import reconstruct, measures, csvsave

NAMES=('V6-AL-control','V6-SE','V6-SB')
class SharedV6(v.V6Compact128):
    def __init__(self, mode):
        super().__init__(); self.mode=mode
        if mode!='V6-AL-control':
            # Keep dynamic 5-channel RGB+XY backbone and remove separate static CNN.
            if mode=='V6-SB': self.static_head=self.static_encoder.fc
            del self.static_encoder
    def encode_static(self,x):
        if self.mode=='V6-AL-control': return super().encode_static(x)
        enc=self.dynamic_encoder.encoder
        if self.mode=='V6-SE': return self.dynamic_encoder(x)
        b,_,h,w=x.shape
        xx=torch.linspace(-1,1,w,device=x.device,dtype=x.dtype).view(1,1,1,w).expand(b,1,h,w)
        yy=torch.linspace(-1,1,h,device=x.device,dtype=x.dtype).view(1,1,h,1).expand(b,1,h,w)
        return self.static_head(enc.features(torch.cat((x,xx,yy),1)).flatten(1))
    def forward(self,x,y):
        s=self.encode_static(x); z=self.encode_dynamic(y-x)
        rec=self.decode_static_aux(s)
        o=reconstruct(self,x,z,s,None,True)
        o.update(delta=y-x,static_aux_recon=rec,z_static=s,z_dynamic=z)
        return o

def make_model(mode,ck):
    m=SharedV6(mode)
    sd=dict(ck['model_state_dict'])
    if mode!='V6-AL-control':
        if mode=='V6-SB':
            for k in ('weight','bias'): sd['static_head.'+k]=sd['static_encoder.fc.'+k]
        sd={k:x for k,x in sd.items() if not k.startswith('static_encoder.')}
    m.load_state_dict(sd,strict=True)
    return m.cuda()

def freeze_bn_stats(m):
    # Same policy for all three: pretrained BN running stats fixed; affine trainable.
    for layer in m.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm): layer.eval()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--stride',type=int,default=2); p.add_argument('--encoder-lr',type=float,default=1e-4); p.add_argument('--decoder-lr',type=float,default=3e-4); p.add_argument('--region-weight',type=float,default=4.); p.add_argument('--static-weight',type=float,default=1.); p.add_argument('--smoke',action='store_true'); p.add_argument('--out',type=Path); a=p.parse_args()
    assert min(a.epochs,a.batch_size,a.stride)>0 and a.region_weight>=0 and a.static_weight>=0
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    src=v.cfg.OUT/'v6_residual_ablation'/'V6-AL.pt'
    ck=torch.load(src,map_location='cpu',weights_only=False); assert ck['reconstruction']=='alpha'
    out=a.out or v.cfg.OUT/('v6_shared_encoder_smoke' if a.smoke else 'v6_shared_encoder'); out.mkdir(parents=True,exist_ok=False)
    keys,pair,labels,obj,rgb=v.load_data(); tr,va,te=v.split_pairs(pair,obj)
    for name,ids in [('final_train_pairs',tr),('validation_pairs',va),('test_pairs',te)]: assert list(ids)==list(ck[name]),name
    train,val=[np.flatnonzero(np.isin(pair,q)) for q in (tr,va)]
    assert len(train)==108 and len(val)==18
    def batches(ids,stride,epoch=None):
        ii,tt=np.meshgrid(ids,np.arange(0,223,stride),indexing='ij'); ii=ii.ravel(); tt=tt.ravel(); order=np.arange(len(ii))
        if epoch is not None: np.random.default_rng(42+epoch).shuffle(order)
        if a.smoke: order=order[:2*a.batch_size]
        for j in range(0,len(order),a.batch_size):
            k=order[j:j+a.batch_size]; yield ii[k],tt[k]
    def run(m,epoch=None,opt=None,scaler=None):
        training=opt is not None; m.train(training); freeze_bn_stats(m); sums={}; n=0
        for i,t in batches(train if training else val,a.stride if training else 1,epoch):
            x=v.device_rgb(rgb[i,0]); y=v.device_rgb(rgb[i,t])
            if training: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training),v.amp():
                # Compute static branch once per unique trajectory in this batch.
                ui,inv=np.unique(i,return_inverse=True); inv=torch.as_tensor(inv,device=x.device)
                xs=v.device_rgb(rgb[ui,0]); ss=m.encode_static(xs); rec=m.decode_static_aux(ss)
                mi=(rec.float()-xs).square().flatten(1).mean(1)[inv]
                z=m.encode_dynamic(y-x); o=reconstruct(m,x,z,ss[inv],mi,True); o['delta']=y-x
                losses=v.full_losses(o,x,y); met=measures(o,x,y)
                total=losses['total']+a.region_weight*met['region_loss']+a.static_weight*mi.mean()
            if training: scaler.scale(total).backward(); scaler.step(opt); scaler.update()
            for k,b in dict(met,static_mse=mi.mean(),original_loss=losses['total'],objective=total).items(): sums[k]=sums.get(k,0.)+float(b.detach())*len(i)
            n+=len(i)
        return {k:b/n for k,b in sums.items()}
    def cycle(m):
        m.eval(); vals=[]
        with torch.no_grad(),v.amp():
            for i,t in batches(val,10):
                x=v.device_rgb(rgb[i,0]); y=v.device_rgb(rgb[i,t]); s=m.encode_static(x); z=m.encode_dynamic(y-x); cur=z; errors=[]
                for _ in range(5):
                    o=reconstruct(m,x,cur,s,None,True); cur=m.encode_dynamic(o['frame_recon']-x)
                    errors.append(((cur.float()-z.float()).norm(dim=1)/z.float().norm(dim=1).clamp_min(1e-6)).cpu().numpy())
                vals.append(np.stack(errors,1))
        e=np.concatenate(vals).mean(0); return dict(cycle1_relative_l2=float(e[0]),cycle5_relative_l2=float(e[4]))
    pid=pair[val[0]]; examples=[int(np.flatnonzero((pair==pid)&(labels==l))[0]) for l in (0,1)]; frames=[25,75,125,175]; images={}; hist=[]; summary=[]
    def capture(name,m):
        m.eval(); images[name]=[]
        with torch.no_grad(),v.amp():
            for i in examples:
                x=v.device_rgb(rgb[[i]*4,0]); y=v.device_rgb(rgb[i,frames]); o=m(x,y)
                images[name].append(o['frame_recon'].float().cpu().numpy().transpose(0,2,3,1))
    def counts(m):
        total=sum(p.numel() for p in m.parameters()); aux=sum(p.numel() for p in m.static_aux_decoder.parameters())+sum(p.numel() for p in m.motion_centroid_head.parameters())
        return dict(parameters=total,inference_parameters=total-aux)
    v.seed_everything(42); baseline=make_model(NAMES[0],ck); summary.append(dict(model='V6-AL-original',epoch=0,**counts(baseline),**run(baseline),**cycle(baseline))); capture('V6-AL-original',baseline); del baseline
    csvsave(out/'comparison.csv',summary)
    for name in NAMES:
        v.seed_everything(42); m=make_model(name,ck); print(name,counts(m),flush=True)
        enc=[p for k,p in m.named_parameters() if k.startswith(('static_encoder.','dynamic_encoder.','static_head.'))]; enc_ids={id(p) for p in enc}; dec=[p for p in m.parameters() if id(p) not in enc_ids]
        opt=torch.optim.AdamW([{'params':enc,'lr':a.encoder_lr},{'params':dec,'lr':a.decoder_lr}],weight_decay=1e-5); scaler=torch.amp.GradScaler('cuda'); best=float('inf'); chosen=None
        for ep in range((1 if a.smoke else a.epochs)+1):
            if ep: run(m,ep,opt,scaler)
            r=dict(model=name,epoch=ep,**counts(m),**run(m)); hist.append(r); print(r,flush=True)
            # Common selection criterion, not architecture-specific training objective.
            if r['moving_mse']<best:
                best=r['moving_mse']; chosen=r
                v.atomic_save(dict(model_state_dict=m.state_dict(),architecture='v6_shared_encoder',encoder_mode=name,reconstruction='alpha',static_latent_dim=128,dynamic_latent_dim=128,epoch=ep,final_train_pairs=tr,validation_pairs=va,test_pairs=te),out/(name+'.pt'))
            csvsave(out/'history.csv',hist)
        m.load_state_dict(torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)['model_state_dict'],strict=True); summary.append(dict(chosen,**cycle(m))); capture(name,m); csvsave(out/'comparison.csv',summary); del m,opt,scaler,enc,dec
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for l,label in enumerate(('safe','pain')):
        panels={'Original':rgb[examples[l],frames].astype(np.float32)/255,**{k:x[l] for k,x in images.items()}}
        fig,ax=plt.subplots(len(panels),4,figsize=(9,2.1*len(panels)))
        for r,(name,im) in enumerate(panels.items()):
            for c,t in enumerate(frames):
                ax[r,c].imshow(im[c].clip(0,1)); ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
                if c==0: ax[r,c].set_ylabel(name)
                if r==0: ax[r,c].set_title(f'Frame {t}')
        fig.tight_layout(); fig.savefig(out/f'comparison_{label}.png',dpi=300); plt.close(fig)
    v.write_json(out/'settings.json',dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_epoch=ck['epoch'],arguments={k:str(x) if isinstance(x,Path) else x for k,x in vars(a).items()},selection='validation moving_mse',bn='fixed running statistics; trainable affine',initialization='shared backbone from pretrained dynamic encoder; SB retains pretrained static projection',validation_examples=[keys[i] for i in examples],train_pairs=tr,validation_pairs=va,test_pairs=te))
    print('COMPLETE',out,flush=True)
if __name__=='__main__': main()
