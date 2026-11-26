"""Frozen V6 encoders; 2x2 decoder ablation. Run from repo root."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from CLEAVER_OPTIMIZED.experiments import train_v6_compact128 as v

VARIANTS={'V6-R':(False,False),'V6-RL':(False,True),'V6-A':(True,False),'V6-AL':(True,True)}

def reconstruct(model,x,z,s,mi,alpha):
    d=model.dynamic_decoder
    f=torch.cat((z,s),1)
    h=d.fc(d.fuse(f)).reshape(-1,256,8,8)
    for block in (d.up1,d.up2,d.up3,d.up4,d.up5): h=block(h)
    raw=d.delta_head(h)+d.detail_scale*F.interpolate(d.detail_head(f).reshape(-1,3,64,64),size=(256,256),mode='bilinear',align_corners=False)
    logits=d.mask_head(h).float(); mask=logits.sigmoid()
    candidate=raw.float().sigmoid()-x if alpha else raw.float().tanh()
    delta=mask*candidate
    return dict(candidate_delta=candidate,motion_mask_logits=logits,motion_mask=mask,delta_recon=delta,frame_recon=(x+delta).clamp(0,1),cached_static_mse=mi,motion_centroid=model.motion_centroid_head(z),delta=None)

def region_mean(a,m):
    # Per-image mean over selected pixels and RGB channels; exclude empty regions.
    a=a.float().mean(1,keepdim=True); m=m.float(); n=m.flatten(1).sum(1)
    return ((a*m).flatten(1).sum(1)/n.clamp_min(1))[n>0]

def measures(out,x,y):
    delta=y-x; err=(out['frame_recon']-y).square(); changed=delta.abs().mean(1,keepdim=True)>.035
    dark=delta.mean(1,keepdim=True)<-.035; bright=delta.mean(1,keepdim=True)>.035
    p=out['motion_mask']>.5
    def avg(a): return a.mean() if a.numel() else err.new_tensor(0.)
    return dict(mask_missed_fraction=avg(region_mean((~p).float(),changed)),ungated_candidate_mse=avg(region_mean((out['candidate_delta']-delta).square(),changed)),rgb_mse=err.mean(),moving_mse=avg(region_mean(err,changed)),negative_change_mse=avg(region_mean(err,dark)),positive_change_mse=avg(region_mean(err,bright)),background_mse=avg(region_mean(err,~changed)),mask_iou=((p&changed).flatten(1).sum(1)/(p|changed).flatten(1).sum(1).clamp_min(1)).mean(),edge_l1=(F.l1_loss(out['frame_recon'][:,:,:,1:]-out['frame_recon'][:,:,:,:-1],y[:,:,:,1:]-y[:,:,:,:-1])+F.l1_loss(out['frame_recon'][:,:,1:,:]-out['frame_recon'][:,:,:-1,:],y[:,:,1:,:]-y[:,:,:-1,:])),region_loss=avg(region_mean(err,changed)))

def csvsave(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--stride',type=int,default=2); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--region-weight',type=float,default=4.); p.add_argument('--smoke',action='store_true'); p.add_argument('--out',type=Path); a=p.parse_args()
    assert min(a.epochs,a.batch_size,a.stride)>0 and a.region_weight>=0
    root=v.cfg.OUT; src=root/'autoencoder_v6compact128'/'autoencoder.pt'; out=a.out or root/('v6_residual_ablation_smoke' if a.smoke else 'v6_residual_ablation'); out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(src,map_location='cpu',weights_only=False); keys,pair,labels,obj,rgb=v.load_data(); tr,va,te=v.split_pairs(pair,obj)
    assert list(tr)==list(ck['final_train_pairs']) and list(va)==list(ck['validation_pairs']) and list(te)==list(ck['test_pairs'])
    train,val=[np.flatnonzero(np.isin(pair,q)) for q in (tr,va)]; assert len(train)==108 and len(val)==18
    v.seed_everything(42); base=v.V6Compact128().cuda(); base.load_state_dict(ck['model_state_dict']); base.requires_grad_(False).eval()
    s,mi=v.cache_static(base,rgb,np.r_[train,val]); z=torch.empty((len(keys),223,128),device='cuda',dtype=torch.float16)
    with torch.no_grad():
        for i in np.r_[train,val]:
            x=v.device_rgb(rgb[i,0:1]);
            for t in range(0,223,a.batch_size):
                y=v.device_rgb(rgb[i,t:t+a.batch_size]);
                with v.amp(): z[i,t:t+len(y)]=base.encode_dynamic(y-x)
    def batches(ids,stride,epoch):
        ii,tt=np.meshgrid(ids,np.arange(0,223,stride),indexing='ij'); ii=ii.ravel(); tt=tt.ravel(); order=np.arange(len(ii))
        if epoch is not None: np.random.default_rng(42+epoch).shuffle(order)
        if a.smoke: order=order[:a.batch_size*2]
        for j in range(0,len(order),a.batch_size):
            k=order[j:j+a.batch_size]; yield ii[k],tt[k]
    def evaluate(model,alpha,extra,epoch=None,opt=None):
        training=opt is not None; model.eval(); model.dynamic_decoder.train(training); sums={}; n=0
        for i,t in batches(train if training else val,a.stride if training else 1,epoch):
            x=v.device_rgb(rgb[i,0]); y=v.device_rgb(rgb[i,t]);
            if training: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training),v.amp():
                o=reconstruct(model,x,z[i,t],s[i],mi[i],alpha); o['delta']=y-x
                loss=v.full_losses(o,x,y)['total']; m=measures(o,x,y); total=loss+(a.region_weight*m['region_loss'] if extra else 0)
            if training: scaler.scale(total).backward(); scaler.step(opt); scaler.update()
            for k,b in dict(m,original_loss=loss,training_objective=total).items(): sums[k]=sums.get(k,0.)+float(b.detach())*len(i)
            n+=len(i)
        return {k:b/n for k,b in sums.items()}
    def cycle(model,alpha):
        vals=[]
        with torch.no_grad(),v.amp():
            for i,t in batches(val,10,None):
                x=v.device_rgb(rgb[i,0]); zz=z[i,t]; current=zz; errors=[]
                for _ in range(5):
                    o=reconstruct(model,x,current,s[i],mi[i],alpha); current=model.encode_dynamic(o['frame_recon']-x)
                    errors.append(((current.float()-zz.float()).norm(dim=1)/zz.float().norm(dim=1).clamp_min(1e-6)).cpu().numpy())
                vals.append(np.stack(errors,1))
        e=np.concatenate(vals).mean(0); return {'cycle1_relative_l2':float(e[0]),'cycle5_relative_l2':float(e[4])}
    hist=[]; summary=[]; images={}; frames=[25,75,125,175]; pid=pair[val[0]]
    example=[int(np.flatnonzero((pair==pid)&(labels==l))[0]) for l in (0,1)]
    def capture(name,model,alpha):
        model.eval(); images[name]=[]
        with torch.no_grad(),v.amp():
            for i in example:
                x=v.device_rgb(rgb[[i]*4,0]); y=v.device_rgb(rgb[i,frames]); o=reconstruct(model,x,z[i,frames],s[[i]*4],mi[[i]*4],alpha)
                images[name].append(o['frame_recon'].cpu().numpy().transpose(0,2,3,1))
    def row(name,epoch,m): return dict(model=name,epoch=epoch,**m)
    m=evaluate(base,False,False); summary.append(dict(row('V6-base',0,m),**cycle(base,False))); capture('V6-base',base,False)
    for name,(alpha,extra) in VARIANTS.items():
        model=v.V6Compact128().cuda(); model.load_state_dict(ck['model_state_dict']); model.requires_grad_(False).eval(); model.dynamic_decoder.requires_grad_(True)
        opt=torch.optim.AdamW(model.dynamic_decoder.parameters(),lr=a.lr,weight_decay=1e-5); scaler=torch.amp.GradScaler('cuda'); best=float('inf'); chosen=None
        for ep in range(0, (1 if a.smoke else a.epochs)+1):
            if ep: evaluate(model,alpha,extra,ep,opt)
            m=evaluate(model,alpha,extra); r=row(name,ep,m); hist.append(r); print(r,flush=True)
            # Same selection criterion for every variant; region MSE, not incomparable training losses.
            if m['moving_mse']<best:
                best=m['moving_mse']; chosen=r; v.atomic_save(dict(model_state_dict=model.state_dict(),architecture='v6compact128',reconstruction='alpha' if alpha else 'residual',region_loss=extra,epoch=ep,source_epoch=ck['epoch'],final_train_pairs=tr,validation_pairs=va,test_pairs=te),out/(name+'.pt'))
            csvsave(out/'history.csv',hist)
        model.load_state_dict(torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)['model_state_dict']); model.eval(); summary.append(dict(chosen,**cycle(model,alpha))); capture(name,model,alpha); csvsave(out/'comparison.csv',summary); del model,opt,scaler
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for l,label in enumerate(('safe','pain')):
        panels={'Original':rgb[example[l],frames].astype(np.float32)/255,**{k:v[l] for k,v in images.items()}}
        fig,ax=plt.subplots(len(panels),4,figsize=(9,2.1*len(panels)))
        for r,(name,im) in enumerate(panels.items()):
            for c,t in enumerate(frames):
                ax[r,c].imshow(im[c].clip(0,1)); ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
                if c==0: ax[r,c].set_ylabel(name)
                if r==0: ax[r,c].set_title(f'Frame {t}')
        fig.tight_layout(); fig.savefig(out/f'comparison_{label}.png',dpi=300); plt.close(fig)
    v.write_json(out/'settings.json',dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_epoch=ck['epoch'],arguments={k:str(x) if isinstance(x,Path) else x for k,x in vars(a).items()},selection='validation moving_mse',validation_examples=[keys[i] for i in example],cycle_stride=10,parameters=sum(p.numel() for p in base.parameters()),train_pairs=tr,validation_pairs=va,test_pairs=te))
    print('COMPLETE',out,flush=True)

if __name__=='__main__': main()
