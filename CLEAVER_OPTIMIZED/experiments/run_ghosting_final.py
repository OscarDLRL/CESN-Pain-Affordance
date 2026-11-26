"""Final first-frame diagnostics. Run from repository root; validation only."""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
import h5py
import torch
from torch import nn
import torch.nn.functional as F
from CLEAVER_OPTIMIZED.experiments.run_first_frame_limit import Repair, metrics, regions, region_error
from CLEAVER_OPTIMIZED.autoencoder.lightweight_variable import LightweightAutoencoderVariable
from CLEAVER_OPTIMIZED.autoencoder.training_utils import full_losses

class FinalRepair(Repair):
    def __init__(self, base, beta=1., joint=False):
        super().__init__(base,False,True,True)
        self.beta=beta; self.joint=joint
        self.base.dynamic_encoder.requires_grad_(joint)
    def train(self, mode=True):
        super().train(mode)
        # Keep BN running statistics fixed even during joint tuning.
        self.base.dynamic_encoder.eval()
        return self
    def forward(self,s,z,x,y,sm):
        if self.joint: z=self.base.encode_dynamic(y-x)
        h=self.base.dynamic_decoder(self.base.fuse(torch.cat((z,s),1)))
        h=h+self.refine(torch.cat((h,x),1))
        logits=h[:,3:]; mask=(self.beta*logits).sigmoid()
        generated=h[:,:3].sigmoid(); residual=mask*(generated-x)
        return dict(delta=y-x,delta_recon=residual,frame_recon=x+residual,motion_mask_logits=self.beta*logits,motion_mask=mask,motion_centroid=self.base.motion_centroid_head(z),cached_static_mse=sm)

def objective(o,x,y,hard):
    loss=full_losses(o,x,y)['total']
    e=(o['frame_recon'].float()-y).square().mean(1,keepdim=True)
    m,neg=regions(x,y)
    loss=loss+2*region_error(e,m)+2*region_error(e,neg)
    if hard:
        # RGB change support, not semantic segmentation. Dilate by 2 pixels.
        target=F.max_pool2d(((y-x).abs().mean(1,keepdim=True)>.035).float(),5,1,2)
        logits=o['motion_mask_logits'].float()
        bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
        loss=loss+region_error(bce,target)+.25*region_error(bce,1-target)
    return loss

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['sharp','mask','joint'],required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--stride',type=int,default=2); p.add_argument('--beta',type=float,default=2); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    assert min(a.epochs,a.batch_size,a.stride,a.beta)>0
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.manual_seed(42)
    root=Path('CLEAVER_OPTIMIZED/outputs/final162'); ae=root/'autoencoder_lightweight128_matched'; out=root/('ghosting_final_'+a.stage+('_smoke' if a.smoke else '')); out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.csv').exists(): raise RuntimeError('Output already exists; rename it before rerunning: '+str(out))
    base=LightweightAutoencoderVariable(128).cuda(); ck=torch.load(ae/'autoencoder.pt',map_location='cpu',weights_only=False); base.load_state_dict(ck['model_state_dict']); base.eval().requires_grad_(False)
    source=root/'first_frame_limit/AE08.pt'; repair=torch.load(source,map_location='cpu',weights_only=False); assert repair['alpha'] and repair['refine'] and not repair['detail']
    split=json.loads((ae/'split.json').read_text()); assert [len(split[k]) for k in ('train','validation','test')]==[54,9,18]; assert len(set().union(*map(set,split.values())))==81
    data=Path('datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5')
    # Keep only needed trajectories; sharp diagnostic does not load training RGB.
    wanted=set(split['validation']+(split['train'] if a.stage!='sharp' else []))
    with h5py.File(data,'r') as f,h5py.File(ae/'cleaver_lightweight_latent128.h5','r') as l:
        keys=[k for k in sorted(f) if k.startswith('trajectory_') and int(f[k].attrs['pair_id']) in wanted]; assert len(keys)==(18 if a.stage=='sharp' else 126)
        rgb=np.empty((len(keys),223,256,256,3),np.uint8); ss=[]; zz=[]; ids=[]; labels=[]
        for i,k in enumerate(keys):
            g=f[k]; q=l[k]; assert int(g.attrs['pair_id'])==int(q.attrs['pair_id']) and int(g.attrs['label'])==int(q.attrs['label']); g['rgb'].read_direct(rgb[i]); ss.append(q['z_static'][:]); zz.append(q['z_dynamic'][:]); ids.append(int(g.attrs['pair_id'])); labels.append(int(g.attrs['label']))
    s=torch.tensor(np.array(ss),device='cuda'); z=torch.tensor(np.array(zz),device='cuda'); assert s.shape==(len(keys),128) and z.shape==(len(keys),223,128)
    def image(v): return torch.from_numpy(np.ascontiguousarray(v)).cuda().permute(0,3,1,2).float()/255
    sm=[]
    with torch.no_grad():
        for j in range(0,len(keys),16):
            x=image(rgb[j:j+16,0]); y=image(rgb[j:j+16,75])
            with torch.autocast('cuda',dtype=torch.float16): st=base.encode_static(x); dy=base.encode_dynamic(y-x); rec=base.decode_static_aux(st)
            torch.testing.assert_close(st.float(),s[j:j+16],rtol=.02,atol=.02); torch.testing.assert_close(dy.float(),z[j:j+16,75],rtol=.02,atol=.02); sm.append((rec.float()-x).square().flatten(1).mean(1))
    sm=torch.cat(sm); train=np.flatnonzero(np.isin(ids,split['train'])); val=np.flatnonzero(np.isin(ids,split['validation']))
    frames=np.unique(np.r_[np.arange(0,223,a.stride),222]); vf=np.arange(0,223,4); ts=[25,75,125,175]; pid=ids[val[0]]; examples=[next(i for i in val if ids[i]==pid and labels[i]==lab) for lab in (0,1)]
    history=[]; rows=[]; pictures={}; scaler=torch.amp.GradScaler('cuda')
    def iterate(model,indices,fs,opt=None,ep=0,hard=False):
        training=opt is not None; model.train(training); pairs=np.array([(i,t) for i in indices for t in fs]);
        if training: np.random.default_rng(42+ep).shuffle(pairs)
        if a.smoke: pairs=pairs[:2*a.batch_size]
        sums={}; n=0
        for start in range(0,len(pairs),a.batch_size):
            i,t=pairs[start:start+a.batch_size].T; x=image(rgb[i,0]); y=image(rgb[i,t])
            if training: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast('cuda',dtype=torch.float16): o=model(s[i],z[i,t],x,y,sm[i]); loss=objective(o,x,y,hard)
                if training: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad(): ms=metrics(o,x,y); ms['loss']=loss.detach()
            for k,v in ms.items(): sums[k]=sums.get(k,0)+float(v)*len(i)
            n+=len(i)
        r={k:v/n for k,v in sums.items()}; assert all(np.isfinite(v) for v in r.values()); return r
    def capture(model,name):
        model.eval(); ims=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for i in examples:
                x=image(np.repeat(rgb[i,0:1],4,axis=0)); y=image(rgb[i,ts]); o=model(s[i:i+1].expand(4,-1),z[i,ts],x,y,sm[i:i+1].expand(4)); ims.append(o['frame_recon'].float().cpu().permute(0,2,3,1).numpy())
        pictures[name]=ims
    def make(beta,joint=False):
        model=FinalRepair(base,beta,joint).cuda(); result=model.load_state_dict(repair['decoder_state'],strict=False); assert not result.unexpected_keys; assert set(repair['decoder_state'])=={k for k in model.state_dict() if k.startswith(('base.fuse.','base.dynamic_decoder.','refine.'))}; return model
    def record(name,ep,m):
        history.append(dict(variant=name,epoch=ep,**m)); print(name,ep,m,flush=True)
    for beta in ([1,2,4,8,16] if a.stage=='sharp' else [1,a.beta]):
        name=f'AE09_beta{beta:g}'; model=make(beta); m=iterate(model,val,vf); record(name,0,m); rows.append(dict(variant=name,best_epoch=0,**m)); capture(model,name); del model
    if a.stage!='sharp':
        # Both AE10 and AE11 start from identical AE08 weights and beta.
        # AE11 is therefore a matched encoder-unfreezing ablation of AE10.
        joint=a.stage=='joint'; name='AE11' if joint else 'AE10'; model=make(a.beta,joint)
        enc=list(model.base.dynamic_encoder.parameters()); enc_ids={id(q) for q in enc}; dec=[q for q in model.parameters() if q.requires_grad and id(q) not in enc_ids]
        groups=[dict(params=dec,lr=3e-4)]
        if joint: groups.append(dict(params=enc,lr=1e-4))
        opt=torch.optim.AdamW(groups,weight_decay=1e-5); best=float('inf'); state=None
        for ep in range((1 if a.smoke else a.epochs)+1):
            if ep: iterate(model,train,frames,opt,ep,True)
            m=iterate(model,val,vf,hard=True); record(name,ep,m); score=m['moving_mse']+m['negative_change_mse']
            if score<best:
                best=score; bm=m; be=ep; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state); capture(model,name); rows.append(dict(variant=name,best_epoch=be,**bm)); torch.save(dict(model_state_dict=state,beta=a.beta,joint=joint,alpha=True,refine=True,detail=False,best_epoch=be,source=str(source)),out/f'{name}.pt')
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
    (out/'settings.json').write_text(json.dumps(dict(vars(a),split=split,source=str(source),selection='moving_mse + negative_change_mse',examples=[keys[i] for i in examples],mask_target='dilated RGB difference proxy; not segmentation',joint_requires_new_latents= a.stage=='joint'),indent=2)); print('COMPLETE',out,flush=True)
if __name__=='__main__': main()
