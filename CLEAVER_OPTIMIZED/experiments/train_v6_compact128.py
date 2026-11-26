
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvEncoder256(nn.Module):
    """
    Native 256x256 image -> latent.
    Five stride-2 stages:
      256 -> 128 -> 64 -> 32 -> 16 -> 8
    """

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        self.fc = nn.Linear(
            256 * 8 * 8,
            int(latent_dim),
        )

    def forward(self, x):
        h = self.features(x)
        return self.fc(h.flatten(1))


class CoordDynamicEncoder256(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder256(
            in_channels=5,
            latent_dim=latent_dim,
        )

    def forward(self, delta):
        B, _, H, W = delta.shape

        x = torch.linspace(
            -1.0, 1.0, W,
            device=delta.device,
            dtype=delta.dtype,
        ).view(1, 1, 1, W).expand(B, 1, H, W)

        y = torch.linspace(
            -1.0, 1.0, H,
            device=delta.device,
            dtype=delta.dtype,
        ).view(1, 1, H, 1).expand(B, 1, H, W)

        return self.encoder(
            torch.cat([delta, x, y], dim=1)
        )


class StaticAuxDecoder256(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(
            int(latent_dim),
            256 * 8 * 8,
        )

        self.up1 = UpBlock(256, 128)  # 8->16
        self.up2 = UpBlock(128, 64)   # 16->32
        self.up3 = UpBlock(64, 32)    # 32->64
        self.up4 = UpBlock(32, 16)    # 64->128
        self.up5 = UpBlock(16, 16)    # 128->256

        self.rgb_head = nn.Conv2d(
            16, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        return torch.sigmoid(self.rgb_head(h))


class MotionGatedDynamicDecoder256(nn.Module):
    """
    Native 256x256 residual decoder.

    The main decoder is truly 256-native.
    The small direct-detail branch remains 64x64 internally and is
    upsampled before fusion so parameter count does not explode.
    """

    def __init__(
        self,
        static_latent_dim=128,
        dynamic_latent_dim=128,
    ):
        super().__init__()

        fused_dim = (
            int(static_latent_dim)
            + int(dynamic_latent_dim)
        )

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.SiLU(inplace=True),
            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
        )

        self.fc = nn.Linear(
            256,
            256 * 8 * 8,
        )

        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.up5 = UpBlock(16, 16)

        self.delta_head = nn.Conv2d(
            16, 3, kernel_size=3, padding=1
        )

        self.mask_head = nn.Conv2d(
            16, 1, kernel_size=3, padding=1
        )

        # Keep parameter count comparable to baseline.
        self.detail_head = nn.Linear(
            fused_dim,
            3 * 64 * 64,
        )

        self.detail_scale = nn.Parameter(
            torch.tensor(0.05)
        )

    def forward(self, z_dynamic, z_static):
        fused = torch.cat(
            [z_dynamic, z_static],
            dim=1,
        )

        h = self.fuse(fused)
        h = self.fc(h).view(
            -1, 256, 8, 8
        )

        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)

        coarse_delta = self.delta_head(h)

        detail = self.detail_head(
            fused
        ).view(
            -1, 3, 64, 64
        )

        detail = F.interpolate(
            detail,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        candidate_delta = torch.tanh(
            coarse_delta
            + self.detail_scale * detail
        )

        motion_mask_logits = self.mask_head(h)
        motion_mask = torch.sigmoid(
            motion_mask_logits
        )

        delta_recon = (
            candidate_delta
            * motion_mask
        )

        return {
            "candidate_delta": candidate_delta,
            "motion_mask_logits": motion_mask_logits,
            "motion_mask": motion_mask,
            "delta_recon": delta_recon,
        }


class V6Compact128(nn.Module):
    """
    V6 native-256, no future anchor.

    Input/output image resolution: 256x256.
    Latents:
      z_static  = 128
      z_dynamic = 128

    No task, grasp XY, qpos, TCP or future frames enter the encoder.
    """

    def __init__(
        self,
        static_latent_dim=128,
        dynamic_latent_dim=128,
    ):
        super().__init__()

        self.static_latent_dim = int(
            static_latent_dim
        )
        self.dynamic_latent_dim = int(
            dynamic_latent_dim
        )

        self.static_encoder = ConvEncoder256(
            in_channels=3,
            latent_dim=self.static_latent_dim,
        )

        self.dynamic_encoder = CoordDynamicEncoder256(
            latent_dim=self.dynamic_latent_dim,
        )

        self.static_aux_decoder = StaticAuxDecoder256(
            latent_dim=self.static_latent_dim,
        )

        self.dynamic_decoder = MotionGatedDynamicDecoder256(
            static_latent_dim=self.static_latent_dim,
            dynamic_latent_dim=self.dynamic_latent_dim,
        )

        self.motion_centroid_head = nn.Sequential(
            nn.Linear(self.dynamic_latent_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def encode_static(self, frame0):
        return self.static_encoder(frame0)

    def encode_dynamic(self, delta):
        return self.dynamic_encoder(delta)

    def decode_static_aux(self, z_static):
        return self.static_aux_decoder(z_static)

    def decode_dynamic_full(
        self,
        z_dynamic,
        z_static,
    ):
        return self.dynamic_decoder(
            z_dynamic,
            z_static,
        )

    def decode_dynamic(
        self,
        z_dynamic,
        z_static,
    ):
        return self.decode_dynamic_full(
            z_dynamic,
            z_static,
        )["delta_recon"]

    def forward(self, frame0, frame_t, cached_z_static=None, cached_static_mse=None):
        delta = frame_t - frame0

        z_static = self.encode_static(frame0) if cached_z_static is None else cached_z_static
        z_dynamic = self.encode_dynamic(delta)

        static_aux_recon = (
            self.decode_static_aux(z_static) if cached_static_mse is None else None
        )

        dynamic_outputs = (
            self.decode_dynamic_full(
                z_dynamic,
                z_static,
            )
        )

        delta_recon = dynamic_outputs[
            "delta_recon"
        ]

        frame_recon = torch.clamp(
            frame0 + delta_recon,
            0.0,
            1.0,
        )

        motion_centroid = (
            self.motion_centroid_head(
                z_dynamic
            )
        )

        return {
            "z_static": z_static,
            "z_dynamic": z_dynamic,
            "delta": delta,
            "static_aux_recon":
                static_aux_recon,
            "cached_static_mse": cached_static_mse,
            "candidate_delta":
                dynamic_outputs[
                    "candidate_delta"
                ],
            "motion_mask_logits":
                dynamic_outputs[
                    "motion_mask_logits"
                ],
            "motion_mask":
                dynamic_outputs[
                    "motion_mask"
                ],
            "delta_recon": delta_recon,
            "frame_recon": frame_recon,
            "motion_centroid":
                motion_centroid,
        }

"""Native RGB256 lightweight AE. Separate outputs; no test-based early stopping."""
import argparse, json, time, hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from CLEAVER_OPTIMIZED import config as cfg

from CLEAVER_OPTIMIZED.autoencoder.training_utils import full_losses, seed_everything, motion_soft_mask

OUT=cfg.OUT/'autoencoder_v6compact128'
LAT=OUT/'cleaver_v6compact_latent128.h5'

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
        assert set(pair)==set(cfg.TRAIN)|set(cfg.SPLIT["validation"])|set(cfg.TEST)
        assert not set(cfg.TRAIN)&set(cfg.TEST)
        for p in set(pair): assert sorted(labels[pair==p].tolist())==[0,1]
        # Allocate once, retain uint8, never create a second full float32 copy.
        rgb=np.empty((len(keys),223,256,256,3),dtype=np.uint8)
        for i,k in enumerate(keys):
            ds=f[k]['rgb']; assert ds.shape==rgb[i].shape and ds.dtype==np.uint8
            ds.read_direct(rgb[i])
    print(f'RGB cache: {rgb.nbytes/2**30:.2f} GiB | {len(keys)} trajectories',flush=True)
    return keys, pair, labels, obj, rgb

# FINAL162_FIXED_SPLIT

def split_pairs(pair,obj):
    return tuple(sorted(map(int,cfg.SPLIT[k])) for k in ('train','validation','test'))

def device_rgb(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to('cuda').permute(0,3,1,2).float().div_(255)

def batches(ids,T,batch,rng=None):
    ix=np.arange(len(ids)*T)
    if rng is not None: ix=ix[(ix%T)%TRAIN_STRIDE==0]
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

def export(model,keys,rgb,labels,pair,ids_test,batch, skip_latents=False):
    model.eval(); tmp=LAT.with_suffix('.tmp')
    if not skip_latents:
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
        fig.tight_layout(); fig.savefig(OUT/f'autoencoder_{name}.png',dpi=300); plt.close(fig)

def main():
    global OUT,LAT,TRAIN_STRIDE
    ap=argparse.ArgumentParser(); ap.add_argument('--static-epochs',type=int,default=15); ap.add_argument('--joint-epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--static-batch',type=int,default=16); ap.add_argument('--patience',type=int,default=5); ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--resume',action='store_true'); ap.add_argument("--dynamic-dim",type=int,choices=[128],default=128); ap.add_argument("--stride",type=int,default=2); ap.add_argument("--skip-export",action="store_true"); args=ap.parse_args(); TRAIN_STRIDE=args.stride; assert TRAIN_STRIDE>0
    OUT=cfg.OUT/f"autoencoder_v6compact{args.dynamic_dim}"; LAT=OUT/f"cleaver_v6compact_latent{args.dynamic_dim}.h5"
    assert min(args.static_epochs,args.joint_epochs,args.batch_size,args.static_batch,args.patience)>0
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    seed_everything(42); OUT.mkdir(parents=True,exist_ok=True)
    latest=OUT/'latest.pt'; best=OUT/'autoencoder.pt'
    if not args.resume and (latest.exists() or best.exists()): raise RuntimeError('Existing checkpoints: use --resume or archive this output folder')
    keys,pair,labels,obj,rgb=load_data(); tr,va,te=split_pairs(pair,obj)
    ids=[np.flatnonzero(np.isin(pair,s)) for s in (tr,va,te)]; train,val,test=ids
    split=dict(train=tr,validation=va,test=te); write_json(OUT/'split.json',split); print('Pairs:',{k:len(v) for k,v in split.items()},flush=True)
    # Identity check prevents resume against changed data or split.
    stat=cfg.RGB.stat(); signature=dict(dataset=str(cfg.RGB.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,split=split,static_epochs=args.static_epochs,batch=args.batch_size,static_batch=args.static_batch,lr=args.lr)
    model=V6Compact128(128, args.dynamic_dim).cuda(); print('Parameters:',sum(p.numel() for p in model.parameters()),flush=True)
    scaler=torch.amp.GradScaler('cuda'); rng=np.random.default_rng(42); static_done=0; joint_done=0; best_loss=float('inf'); stale=0; history=[]
    resume=None
    signature.update(dynamic_dim=args.dynamic_dim,static_training='from_scratch',stride=TRAIN_STRIDE)
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
            best_loss=vm['total']; stale=0; atomic_save(dict(model_state_dict=model.state_dict(),architecture='v6compact128',static_latent_dim=128,dynamic_latent_dim=args.dynamic_dim,final_train_pairs=tr,validation_pairs=va,test_pairs=te,epoch=epoch),best)
        else: stale+=1
        save(opt,'joint'); write_json(OUT/'history.json',history)
        print(f'JOINT {epoch:03d} train={tm["total"]:.6f} val={vm["total"]:.6f} sec={history[-1]["seconds"]:.1f} patience={stale}/{args.patience}',flush=True)
    checkpoint=torch.load(best,map_location='cpu',weights_only=False); model.load_state_dict(checkpoint['model_state_dict']); cache=cache_static(model,rgb,val)
    metrics=run_epoch(model,rgb,val,cache,args.batch_size); write_json(OUT/'training_summary.json',dict(split=split,best_epoch=checkpoint['epoch'],validation_metrics=metrics,history=history,parameters=sum(p.numel() for p in model.parameters())))
    print('VALIDATION',metrics,flush=True); export(model,keys,rgb,labels,pair,val,args.batch_size,args.skip_export); print('DONE',OUT,flush=True)



if __name__=="__main__": main()
