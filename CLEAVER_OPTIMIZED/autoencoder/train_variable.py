"""Native RGB256 lightweight AE. Separate outputs; no test-based early stopping."""
import argparse, json, time, hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.autoencoder.lightweight_variable import LightweightAutoencoderVariable
from CLEAVER_OPTIMIZED.autoencoder.training_utils import full_losses, seed_everything, motion_soft_mask

OUT=cfg.OUT/'autoencoder_lightweight128'
LAT=OUT/'cleaver_lightweight_latent128.h5'

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def write_json(path,obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)

def amp(): return torch.autocast('cuda',dtype=torch.float16)

def load_data():
    with h5py.File(cfg.RGB,'r') as f:
        keys=sorted(k for k in f if k.startswith('trajectory_'))
        pair=np.array([int(f[k].attrs['pair_id']) for k in keys])
        labels=np.array([int(f[k].attrs['label']) for k in keys])
        obj=np.array([int(f[k].attrs['object_id']) for k in keys])
        assert set(pair)==set(cfg.TRAIN)|set(cfg.TEST)
        assert not set(cfg.TRAIN)&set(cfg.TEST)
        for p in set(pair): assert sorted(labels[pair==p].tolist())==[0,1]
        # Allocate once, retain uint8, never create a second full float32 copy.
        rgb=np.empty((len(keys),223,256,256,3),dtype=np.uint8)
        for i,k in enumerate(keys):
            ds=f[k]['rgb']; assert ds.shape==rgb[i].shape and ds.dtype==np.uint8
            ds.read_direct(rgb[i])
    print(f'RGB cache: {rgb.nbytes/2**30:.2f} GiB | {len(keys)} trajectories',flush=True)
    return keys, pair, labels, obj, rgb

def split_pairs(pair,obj):
    # Keep test fixed. Hold out one training pair per distinct object where possible.
    rng=np.random.default_rng(42); train=set(map(int,cfg.TRAIN)); val=[]
    groups={int(o):sorted({int(p) for p in pair[obj==o] if int(p) in train}) for o in np.unique(obj)}
    for o,ps in sorted(groups.items()):
        if len(ps)>1: val.append(int(rng.choice(ps)))
    if not val: raise RuntimeError('Cannot create grouped validation split')
    return sorted(train-set(val)), sorted(val), sorted(map(int,cfg.TEST))

def device_rgb(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to('cuda').permute(0,3,1,2).float().div_(255)

def batches(ids,T,batch,rng=None):
    ix=np.arange(len(ids)*T)
    if rng is not None: rng.shuffle(ix)
    for i in range(0,len(ix),batch):
        a=ix[i:i+batch]; yield ids[a//T],a%T

def cache_static(model,rgb,ids):
    model.eval(); z=torch.zeros((len(rgb),128),device='cuda',dtype=torch.float16); mse=torch.zeros(len(rgb),device='cuda')
    with torch.no_grad():
        for i in range(0,len(ids),16):
            ix=ids[i:i+16]; x=device_rgb(rgb[ix,0]);
            with amp(): s=model.encode_static(x); rec=model.decode_static_aux(s)
            z[ix]=s; mse[ix]=(rec.float()-x).square().flatten(1).mean(1)
    return z,mse

def run_epoch(model,rgb,ids,cache,batch,optimizer=None,scaler=None,rng=None):
    training=optimizer is not None; model.train(training); model.static_encoder.eval(); model.static_aux_decoder.eval()
    sums={}; count=0
    for ix,t in batches(ids,rgb.shape[1],batch,rng):
        x=device_rgb(rgb[ix,0]); y=device_rgb(rgb[ix,t]); zi,mi=cache
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training),amp():
            out=model(x,y,zi[ix],mi[ix]); losses=full_losses(out,x,y)
        if training:
            scaler.scale(losses['total']).backward(); scaler.step(optimizer); scaler.update()
        for k,v in losses.items(): sums[k]=sums.get(k,0.)+float(v.detach())*len(ix)
        count+=len(ix)
    return {k:v/count for k,v in sums.items()}

def export(model,keys,rgb,labels,pair,ids_test,batch):
    model.eval(); tmp=LAT.with_suffix('.tmp')
    with torch.no_grad(),h5py.File(cfg.RGB,'r') as src,h5py.File(tmp,'w') as dst:
        for i,k in enumerate(keys):
            x=device_rgb(rgb[i,0:1]); chunks=[]
            with amp(): z=model.encode_static(x)
            for t in range(0,rgb.shape[1],batch):
                y=device_rgb(rgb[i,t:t+batch]);
                with amp(): d=model.encode_dynamic(y-x)
                chunks.append(d.float().cpu().numpy())
            g=dst.create_group(k)
            for a,v in src[k].attrs.items(): g.attrs[a]=v
            g.attrs['pain_label']=int(labels[i]); g.attrs['native_image_size']=256
            g.create_dataset('z_static',data=z[0].float().cpu().numpy())
            g.create_dataset('z_dynamic',data=np.concatenate(chunks),compression='gzip')
            for name in ('target_world','prepared_tcp','knife_xy','start_offset_xy','contact_impulses'):
                if name in src[k]: g.create_dataset(name,data=src[k][name][()])
            print(f'LATENT {i+1}/{len(keys)} {k}',flush=True)
    tmp.replace(LAT)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pid=pair[ids_test[0]]
    for label,name in [(0,'safe'),(1,'pain')]:
        i=int(np.flatnonzero((pair==pid)&(labels==label))[0]); ts=[25,75,125,175]
        x=device_rgb(np.repeat(rgb[i,0:1],len(ts),axis=0)); y=device_rgb(rgb[i,ts])
        with torch.no_grad(),amp(): out=model(x,y)
        panels=[y,motion_soft_mask(y-x),out['motion_mask'],out['frame_recon']]
        fig,ax=plt.subplots(4,4,figsize=(8,8))
        for r,p in enumerate(panels):
            a=p.float().cpu().numpy().transpose(0,2,3,1)
            for c in range(4):
                ax[r,c].imshow(a[c,:,:,0] if a.shape[-1]==1 else a[c].clip(0,1),cmap='gray' if a.shape[-1]==1 else None,vmin=0,vmax=1)
                ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
                if r==0: ax[r,c].set_title(f'Frame {ts[c]}')
                if c==0: ax[r,c].set_ylabel(['Original RGB','Target mask','Predicted mask','Reconstruction'][r])
        fig.tight_layout(); fig.savefig(OUT/f'autoencoder_{name}.png',dpi=180); plt.close(fig)

def main():
    global OUT,LAT
    ap=argparse.ArgumentParser(); ap.add_argument('--static-epochs',type=int,default=15); ap.add_argument('--joint-epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--static-batch',type=int,default=16); ap.add_argument('--patience',type=int,default=5); ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--resume',action='store_true'); ap.add_argument("--dynamic-dim",type=int,choices=[128,200,256],required=True); args=ap.parse_args()
    OUT=cfg.OUT/f"autoencoder_lightweight{args.dynamic_dim}_matched"; LAT=OUT/f"cleaver_lightweight_latent{args.dynamic_dim}.h5"
    assert min(args.static_epochs,args.joint_epochs,args.batch_size,args.static_batch,args.patience)>0
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    seed_everything(42); OUT.mkdir(parents=True,exist_ok=True)
    latest=OUT/'latest.pt'; best=OUT/'autoencoder.pt'
    if not args.resume and (latest.exists() or best.exists()): raise RuntimeError('Existing checkpoints: use --resume or archive this output folder')
    keys,pair,labels,obj,rgb=load_data(); tr,va,te=split_pairs(pair,obj)
    original=cfg.OUT/"autoencoder_lightweight128"
    saved_split=json.loads((original/"split.json").read_text())
    assert saved_split==dict(train=tr,validation=va,test=te), "Original AE split mismatch"
    ids=[np.flatnonzero(np.isin(pair,s)) for s in (tr,va,te)]; train,val,test=ids
    split=dict(train=tr,validation=va,test=te); write_json(OUT/'split.json',split); print('Pairs:',{k:len(v) for k,v in split.items()},flush=True)
    # Identity check prevents resume against changed data or split.
    stat=cfg.RGB.stat(); signature=dict(dataset=str(cfg.RGB.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,split=split,static_epochs=args.static_epochs,batch=args.batch_size,static_batch=args.static_batch,lr=args.lr)
    model=LightweightAutoencoderVariable(args.dynamic_dim).cuda(); print('Parameters:',sum(p.numel() for p in model.parameters()),flush=True)
    scaler=torch.amp.GradScaler('cuda'); rng=np.random.default_rng(42); static_done=0; joint_done=0; best_loss=float('inf'); stale=0; history=[]
    resume=None
    source=torch.load(original/"autoencoder.pt",map_location="cpu",weights_only=False)["model_state_dict"]
    model.static_encoder.load_state_dict({k.split(".",1)[1]:v for k,v in source.items() if k.startswith("static_encoder.")})
    model.static_aux_decoder.load_state_dict({k.split(".",1)[1]:v for k,v in source.items() if k.startswith("static_aux_decoder.")})
    static_done=args.static_epochs
    signature.update(dynamic_dim=args.dynamic_dim,static_checkpoint_sha256=hashlib.sha256((original/"autoencoder.pt").read_bytes()).hexdigest())
    if args.resume:
        resume=torch.load(latest,map_location='cpu',weights_only=False); assert resume['signature']==signature,'Resume configuration mismatch'
        model.load_state_dict(resume['model_state_dict']); static_done=resume['static_done']; joint_done=resume['joint_done']; best_loss=resume['best_loss']; stale=resume['stale']; history=resume['history']; rng.bit_generator.state=resume['rng']; scaler.load_state_dict(resume['scaler']); torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])
    def save(optimizer,stage):
        atomic_save(dict(model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),stage=stage,static_done=static_done,joint_done=joint_done,best_loss=best_loss,stale=stale,history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),signature=signature),latest)
    opt=torch.optim.AdamW(list(model.static_encoder.parameters())+list(model.static_aux_decoder.parameters()),lr=args.lr,weight_decay=1e-5)
    if resume and resume['stage']=='static': opt.load_state_dict(resume['optimizer'])
    for epoch in range(static_done+1,args.static_epochs+1):
        model.train(); order=rng.permutation(train); total=0.; start=time.perf_counter()
        for i in range(0,len(order),args.static_batch):
            ix=order[i:i+args.static_batch]; x=device_rgb(rgb[ix,0]); opt.zero_grad(set_to_none=True)
            with amp(): loss=F.mse_loss(model.decode_static_aux(model.encode_static(x)),x)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=float(loss.detach())*len(ix)
        static_done=epoch; save(opt,'static'); print(f'STATIC {epoch:03d} loss={total/len(train):.6f} sec={time.perf_counter()-start:.1f}',flush=True)
    for mod in (model.static_encoder,model.static_aux_decoder): mod.requires_grad_(False); mod.eval()
    cache=cache_static(model,rgb,np.concatenate((train,val)))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-5)
    if resume and resume['stage']=='joint': opt.load_state_dict(resume['optimizer'])
    for epoch in range(joint_done+1,args.joint_epochs+1):
        if stale>=args.patience: break
        start=time.perf_counter(); tm=run_epoch(model,rgb,train,cache,args.batch_size,opt,scaler,rng); vm=run_epoch(model,rgb,val,cache,args.batch_size)
        joint_done=epoch; history.append(dict(epoch=epoch,train=tm,validation=vm,seconds=time.perf_counter()-start))
        if not np.isfinite(vm['total']): raise RuntimeError('Nonfinite validation loss')
        if vm['total']<best_loss-1e-5 or not best.exists():
            best_loss=vm['total']; stale=0; atomic_save(dict(model_state_dict=model.state_dict(),architecture='lightweight_variable',static_latent_dim=128,dynamic_latent_dim=args.dynamic_dim,final_train_pairs=tr,validation_pairs=va,test_pairs=te,epoch=epoch),best)
        else: stale+=1
        save(opt,'joint'); write_json(OUT/'history.json',history)
        print(f'JOINT {epoch:03d} train={tm["total"]:.6f} val={vm["total"]:.6f} sec={history[-1]["seconds"]:.1f} patience={stale}/{args.patience}',flush=True)
    checkpoint=torch.load(best,map_location='cpu',weights_only=False); model.load_state_dict(checkpoint['model_state_dict']); cache=cache_static(model,rgb,val)
    metrics=run_epoch(model,rgb,val,cache,args.batch_size); write_json(OUT/'training_summary.json',dict(split=split,best_epoch=checkpoint['epoch'],validation_metrics=metrics,history=history,parameters=sum(p.numel() for p in model.parameters())))
    print('VALIDATION',metrics,flush=True); export(model,keys,rgb,labels,pair,val,args.batch_size); print('DONE',OUT,flush=True)

if __name__=='__main__': main()
