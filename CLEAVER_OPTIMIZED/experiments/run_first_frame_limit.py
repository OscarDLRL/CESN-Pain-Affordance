"""FINAL162 first-frame decoder ablation, stage 2. Frozen encoders; no PDF."""
import argparse, copy, csv, json, math
from pathlib import Path
import numpy as np
import h5py
import torch
from torch import nn
import torch.nn.functional as F
from CLEAVER_OPTIMIZED.autoencoder.lightweight_variable import LightweightAutoencoderVariable
from CLEAVER_OPTIMIZED.autoencoder.training_utils import full_losses

class Repair(nn.Module):
    def __init__(self, base, detail=False, refine=False, alpha=False):
        super().__init__(); self.base=copy.deepcopy(base); self.alpha=alpha
        self.base.requires_grad_(False)
        self.base.fuse.requires_grad_(True); self.base.dynamic_decoder.requires_grad_(True)
        self.detail=nn.Linear(256,3*64*64) if detail else None
        self.refine=nn.Sequential(nn.Conv2d(7,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,4,3,padding=1)) if refine else None
        if self.detail is not None: nn.init.zeros_(self.detail.weight); nn.init.zeros_(self.detail.bias)
        if self.refine is not None: nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
    def forward(self,s,z,x,y,static_mse):
        h=self.base.dynamic_decoder(self.base.fuse(torch.cat((z,s),1)))
        if self.refine is not None: h=h+self.refine(torch.cat((h,x),1))
        delta=h[:,:3]
        if self.detail is not None: delta=delta+F.interpolate(self.detail(torch.cat((z,s),1)).reshape(-1,3,64,64),size=(256,256),mode='bilinear',align_corners=False)
        logits=h[:,3:]; mask=logits.sigmoid()
        if self.alpha:
            generated=delta.sigmoid(); residual=mask*(generated-x); frame=(1-mask)*x+mask*generated
        else:
            residual=delta.tanh()*mask; frame=(x+residual).clamp(0,1)
        return dict(delta=y-x,delta_recon=residual,frame_recon=frame,motion_mask_logits=logits,motion_mask=mask,motion_centroid=self.base.motion_centroid_head(z),cached_static_mse=static_mse)
    def train(self,mode=True):
        super().train(mode)
        self.base.static_encoder.eval(); self.base.dynamic_encoder.eval(); self.base.static_aux_decoder.eval(); self.base.motion_centroid_head.eval()
        return self

# RGB-only proxy: pixels changing appreciably, and pixels losing brightness.
# Not semantic segmentation or a true vacated-region mask.
def regions(x,y):
    d=y-x; return d.abs().mean(1,keepdim=True)>.08, (x-y).mean(1,keepdim=True)>.12

def region_error(e,m):
    return (e*m).sum()/m.sum().clamp_min(1)

def metrics(o,x,y):
    e=(o['frame_recon'].float()-y).square().mean(1,keepdim=True); moving,negative=regions(x,y)
    pred=o['motion_mask']>.5
    edge=(o['frame_recon'][:,:,:,1:]-o['frame_recon'][:,:,:,:-1]-(y[:,:,:,1:]-y[:,:,:,:-1])).abs().mean()+ (o['frame_recon'][:,:,1:,:]-o['frame_recon'][:,:,:-1,:]-(y[:,:,1:,:]-y[:,:,:-1,:])).abs().mean()
    return dict(rgb_mse=e.mean(),moving_mse=region_error(e,moving),negative_change_mse=region_error(e,negative),mask_iou=(pred & moving).sum()/(pred|moving).sum().clamp_min(1),edge_l1=edge)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--stride',type=int,default=2); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--variants',nargs='+',default=['AE04','AE06','AE07','AE08']); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    assert min(a.epochs,a.batch_size,a.stride)>0
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    root=Path('CLEAVER_OPTIMIZED/outputs/final162/autoencoder_lightweight128_matched'); data=Path('datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5'); out=Path('CLEAVER_OPTIMIZED/outputs/final162/first_frame_limit'+('_smoke' if a.smoke else '')); out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.csv').exists(): raise RuntimeError(f'Existing results in {out}; rename the output directory before rerunning')
    split=json.loads((root/'split.json').read_text()); assert [len(split[k]) for k in ('train','validation','test')]==[54,9,18]; assert len(set().union(*map(set,split.values())))==81
    base=LightweightAutoencoderVariable(128).cuda(); checkpoint=torch.load(root/'autoencoder.pt',map_location='cpu',weights_only=False); base.load_state_dict(checkpoint['model_state_dict']); base.eval().requires_grad_(False)
    # Read only training and validation RGB; reuse exact exported latents.
    with h5py.File(data,'r') as f,h5py.File(root/'cleaver_lightweight_latent128.h5','r') as l:
        keys=[k for k in sorted(f) if k.startswith('trajectory_') and int(f[k].attrs['pair_id']) in set(split['train']+split['validation'])]; assert len(keys)==126
        rgb=np.empty((len(keys),223,256,256,3),np.uint8); ss=[]; zz=[]; ids=[]; labels=[]
        for i,k in enumerate(keys):
            g=f[k]; q=l[k]; assert int(g.attrs['pair_id'])==int(q.attrs['pair_id']) and int(g.attrs['label'])==int(q.attrs['label']); g['rgb'].read_direct(rgb[i]); ss.append(q['z_static'][:]); zz.append(q['z_dynamic'][:]); ids.append(int(g.attrs['pair_id'])); labels.append(int(g.attrs['label']))
    s=torch.tensor(np.array(ss),device='cuda'); z=torch.tensor(np.array(zz),device='cuda'); assert s.shape==(126,128) and z.shape==(126,223,128) and torch.isfinite(s).all() and torch.isfinite(z).all()
    def image(v): return torch.from_numpy(np.ascontiguousarray(v)).cuda().permute(0,3,1,2).float()/255
    with torch.no_grad():
        x=image(rgb[:,0]); check=[]; sm=[]
        for j in range(0,len(keys),16):
            with torch.autocast('cuda',dtype=torch.float16):
                st=base.encode_static(x[j:j+16]); dy=base.encode_dynamic(image(rgb[j:j+16,75])-x[j:j+16]); rec=base.decode_static_aux(st)
            torch.testing.assert_close(st.float(),s[j:j+16],rtol=.02,atol=.02); torch.testing.assert_close(dy.float(),z[j:j+16,75],rtol=.02,atol=.02); sm.append((rec.float()-x[j:j+16]).square().flatten(1).mean(1))
        sm=torch.cat(sm); del x
    train=np.flatnonzero(np.isin(ids,split['train'])); val=np.flatnonzero(np.isin(ids,split['validation'])); assert len(train)==108 and len(val)==18
    def iterate(model,indices,frames,optimizer=None,epoch=0):
        training=optimizer is not None; model.train(training); pairs=np.array([(i,t) for i in indices for t in frames]);
        if training: np.random.default_rng(42+epoch).shuffle(pairs)
        if a.smoke: pairs=pairs[:min(len(pairs),a.batch_size*2)]
        sums={}; n=0
        for start in range(0,len(pairs),a.batch_size):
            i,t=pairs[start:start+a.batch_size].T; x=image(rgb[i,0]); y=image(rgb[i,t]);
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast('cuda',dtype=torch.float16): o=model(s[i],z[i,t],x,y,sm[i]); loss=full_losses(o,x,y)['total']
                if getattr(model,'focused',False):
                    e=(o['frame_recon'].float()-y).square().mean(1,keepdim=True); m,neg=regions(x,y); loss=loss+2*region_error(e,m)+2*region_error(e,neg)
                if training: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            with torch.no_grad(): ms=metrics(o,x,y); ms['loss']=loss.detach()
            for k,v in ms.items(): sums[k]=sums.get(k,0)+float(v)*len(i)
            n+=len(i)
        result={k:v/n for k,v in sums.items()}; assert all(math.isfinite(v) for v in result.values()); return result
    variants={'AE00':(False,False,False,False),'AE04':(False,False,True,False),'AE06':(True,False,True,False),'AE07':(False,False,True,True),'AE08':(True,False,True,True)}
    assert all(v in variants and v!='AE00' for v in a.variants)
    rows=[]; pictures={}; history=[]; frames=np.unique(np.r_[np.arange(0,223,a.stride),222]); vf=np.arange(0,223,4); ts=[25,75,125,175]; pid=ids[val[0]]; examples=[next(i for i in val if ids[i]==pid and labels[i]==label) for label in (0,1)]
    assert labels[examples[0]]==0 and labels[examples[1]]==1
    print('Visual examples:', [(keys[i],int(ids[i]),int(labels[i])) for i in examples],flush=True)
    def capture(model,name):
        model.eval(); result=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for i in examples:
                x=image(np.repeat(rgb[i,0:1],4,axis=0)); y=image(rgb[i,ts]); o=model(s[i:i+1].expand(4,-1),z[i,ts],x,y,sm[i:i+1].expand(4)); result.append(o['frame_recon'].float().cpu().permute(0,2,3,1).numpy())
        pictures[name]=result
    for name in ['AE00']+a.variants:
        torch.manual_seed(42); focus,detail,refine,alpha=variants[name]; model=Repair(base,detail,refine,alpha).cuda(); model.focused=focus; opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-5); scaler=torch.amp.GradScaler('cuda'); best=float('inf'); best_state=None; best_metrics=None; best_epoch=0
        for ep in range(0,1 if name=='AE00' else (2 if a.smoke else a.epochs+1)):
            if ep: iterate(model,train,frames,opt,ep)
            m=iterate(model,val,vf); score=m['moving_mse']+m['negative_change_mse']; history.append(dict(variant=name,epoch=ep,**m)); print(name,ep,m,flush=True)
            if score<best: best=score; best_metrics=m; best_epoch=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('base.fuse.','base.dynamic_decoder.','detail.','refine.'))}
        model.load_state_dict(best_state,strict=False); capture(model,name); rows.append(dict(variant=name,best_epoch=best_epoch,trainable_decoder_params=sum(p.numel() for p in model.parameters() if p.requires_grad),score=best,**best_metrics));
        if name!='AE00': torch.save(dict(decoder_state=best_state,variant=name,detail=detail,refine=refine,focused=focus,alpha=alpha,source_checkpoint=str(root/'autoencoder.pt'),best_epoch=best_epoch),out/f'{name}.pt')
        del model,opt,scaler; torch.cuda.empty_cache()
        for fn,records in [('comparison.csv',rows),('history.csv',history)]:
            with (out/fn).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for j,label in enumerate(('safe','pain')):
        fig,ax=plt.subplots(len(pictures)+1,4,figsize=(8,2*(len(pictures)+1)),squeeze=False)
        for r,(name,ims) in enumerate([('Original',rgb[examples[j],ts]/255)]+[(n,v[j]) for n,v in pictures.items()]):
            for c in range(4):
                ax[r,c].imshow(ims[c].clip(0,1)); ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
                if r==0: ax[r,c].set_title(f'Frame {ts[c]}')
                if c==0: ax[r,c].set_ylabel(name)
        fig.tight_layout(); fig.savefig(out/f'comparison_{label}.png',dpi=300); plt.close(fig)
    (out/'settings.json').write_text(json.dumps(dict(vars(a),split=split,selection='moving_mse + negative_change_mse on validation; batch-weighted region metrics',detail_resolution=64,alpha_initialization='same decoder weights, different RGB output mapping; not identical epoch-zero images',reference='first frame'),indent=2)); print('FIRST FRAME LIMIT COMPLETE',out,flush=True)
if __name__=='__main__': main()
