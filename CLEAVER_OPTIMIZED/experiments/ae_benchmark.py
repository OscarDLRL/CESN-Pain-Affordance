"""Six causal visual AE candidates. Run from repository root. No CESN changes."""
import argparse, csv, json, random, time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

ROOT=Path.cwd()
DATA=ROOT/'datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5'
SPLIT=ROOT/'CLEAVER_OPTIMIZED/split.json'
OUT=ROOT/'CLEAVER_OPTIMIZED/outputs/ae_benchmark'
NAMES=['AE01','AE02','AE03','AE04','AE05','AE06']
REF={'AE01','AE02','AE05'}

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def encoder(dim, cin=3):
    layers=[]
    for c in (32,48,64,96,128):
        layers += [nn.Conv2d(cin,c,4,2,1),nn.GroupNorm(8,c),nn.SiLU()]; cin=c
    return nn.Sequential(*layers,nn.Flatten(),nn.Linear(128*8*8,dim))

class Decoder(nn.Module):
    def __init__(self,dim,cout):
        super().__init__(); self.fc=nn.Linear(dim,128*8*8); layers=[]; c=128
        for d in (96,64,48,32,24):
            layers += [nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(c,d,3,padding=1),nn.GroupNorm(8,d),nn.SiLU()]; c=d
        self.net=nn.Sequential(*layers,nn.Conv2d(c,cout,3,padding=1))
    def forward(self,z): return self.net(self.fc(z).reshape(-1,128,8,8))

class Candidate(nn.Module):
    def __init__(self,name):
        super().__init__(); self.name=name
        if name in ('AE01','AE02'):
            from CLEAVER_OPTIMIZED.autoencoder.model import V6DualEncoder
            self.v6=V6DualEncoder(static_latent_dim=128,dynamic_latent_dim=128)
        elif name=='AE03':
            self.enc=encoder(328); self.dec=Decoder(328,3)
        elif name=='AE04':
            self.es=encoder(128); self.ed=encoder(200); self.dec=Decoder(328,3)
        elif name=='AE05':
            self.es=encoder(128); self.ed=encoder(200,6); self.dec=Decoder(328,6)
            y,x=torch.meshgrid(torch.linspace(-1,1,256),torch.linspace(-1,1,256),indexing='ij'); self.register_buffer('grid',torch.stack([x,y],-1)[None])
        elif name=='AE06':
            self.enc=encoder(328); self.bg=Decoder(128,3); self.robot=Decoder(100,4); self.obj=Decoder(100,4)
    def forward(self,r,x):
        if self.name in ('AE01','AE02'):
            o=self.v6(r,x); return o['frame_recon'],o
        if self.name=='AE03': return self.dec(self.enc(x)).sigmoid(),{}
        if self.name=='AE04':
            s=self.es(x); d=self.ed(x); return self.dec(torch.cat([s,d],1)).sigmoid(),{'s':s,'sr':self.es(r)}
        if self.name=='AE05':
            s=self.es(r); d=self.ed(torch.cat([r,x-r],1)); p=self.dec(torch.cat([s,d],1)); flow=p[:,:2].tanh(); alpha=p[:,2:3].sigmoid(); fill=p[:,3:].sigmoid()
            warped=F.grid_sample(r,self.grid+flow.permute(0,2,3,1),align_corners=True,padding_mode='border'); return (1-alpha)*warped+alpha*fill,{'flow':flow,'alpha':alpha}
        z=self.enc(x); b=self.bg(z[:,:128]).sigmoid(); a=self.robot(z[:,128:228]); c=self.obj(z[:,228:]); weights=torch.cat([torch.zeros_like(a[:,:1]),a[:,3:4],c[:,3:4]],1).softmax(1)
        return b*weights[:,:1]+a[:,:3].sigmoid()*weights[:,1:2]+c[:,:3].sigmoid()*weights[:,2:3],{'weights':weights}

class Frames(Dataset):
    def __init__(self,path,keys,name,stride=1,training=False,late=0,gap=0):
        self.path=str(path); self.keys=keys; self.name=name; self.training=training; self.late=late; self.gap=gap; self.f=None
        with h5py.File(path,'r') as f:
            self.items=[(k,t) for k in keys for t in range(max(1,late),len(f[k]['rgb']),stride)]
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        if self.f is None: self.f=h5py.File(self.path,'r')
        k,t=self.items[i]; g=self.f[k]; low=max(1,self.late)
        if self.name=='AE01': r=low
        elif self.training: r=random.randint(1,t)
        elif self.gap: r=max(low,t-self.gap)
        else: r=low
        def get(j): return torch.from_numpy(g['rgb'][j]).permute(2,0,1).contiguous()
        return get(r),get(t),k,t,r

def gradients(x): return x[:,:,:,1:]-x[:,:,:,:-1],x[:,:,1:,:]-x[:,:,:-1,:]
def region_mean(a,w): return (a*w).sum()/ (w.sum()*a.shape[1]+1e-8)
def loss(name,p,x,r,o):
    err=(p-x).abs(); gx,gy=gradients(x); px,py=gradients(p)
    edge=(px-gx).abs().mean()+(py-gy).abs().mean()
    # RGB-only change weighting; not a physical segmentation mask.
    motion=(x-r).abs().amax(1,keepdim=True); w=1+4*(motion>.04).float()
    total=region_mean(err,w)+.25*edge
    if name in ('AE01','AE02'):
        from CLEAVER_OPTIMIZED.autoencoder.losses import full_losses
        # Original V6 objectives plus common image loss, same for AE01/02.
        total=total+full_losses(o,r,x)['total']
    if name=='AE04': total=total+.05*(o['s']-o['sr']).square().mean()+.01*F.relu(.1-o['s'].std(0,unbiased=False)).mean()
    if name=='AE05':
        a,b=gradients(o['flow']); total=total+.01*(a.abs().mean()+b.abs().mean())+.005*o['alpha'].mean()
    return total

def loader(ds,batch,workers,shuffle=False):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=torch.cuda.is_available(),persistent_workers=workers>0)
def convert(a,device): return a.to(device,non_blocking=True).float()/255

def run_epoch(model,dl,device,opt=None,scaler=None):
    model.train(opt is not None); s=0.; n=0
    for r,x,_,_,_ in dl:
        r,x=convert(r,device),convert(x,device)
        if opt: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(opt is not None),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            p,o=model(r,x); l=loss(model.name,p,x,r,o)
        if not torch.isfinite(l): raise RuntimeError('Nonfinite loss')
        if opt: scaler.scale(l).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5); scaler.step(opt); scaler.update()
        s+=l.item()*len(x); n+=len(x)
    return s/max(n,1)

@torch.no_grad()
def evaluate(model,path,keys,device,batch,workers,late=0,gap=0,stride=1):
    model.eval(); ds=Frames(path,keys,model.name,stride,False,late,gap); dl=loader(ds,batch,workers); rows=[]; prev={}
    with h5py.File(path,'r') as f:
        pair9={k for k in keys if int(f[k].attrs['pair_id'])==9} if Path(path)==DATA else set()
        poses={k:f[k]['object_xyz'][:] for k in pair9}
    for r,x,ks,ts,rs in dl:
        r,x=convert(r,device),convert(x,device); p,_=model(r,x); p=p.float().clamp(0,1)
        for i,k in enumerate(ks):
            a,b=p[i:i+1],x[i:i+1]; er=(a-b).abs(); gx,gy=gradients(b); px,py=gradients(a); t=int(ts[i]); old=prev.get(k)
            # Local contrast proxy, not knife/robot segmentation.
            edge=(gx.abs().mean(1,keepdim=True)>.04).float(); edge2=(gy.abs().mean(1,keepdim=True)>.04).float()
            temporal=float('nan'); stationary=float('nan')
            if old is not None and old[0]+stride==t:
                da=a-old[1]; db=b-old[2]; temporal=(da-db).abs().mean().item(); stationary=region_mean(da.abs(),(db.abs().amax(1,keepdim=True)<.01).float()).item()
            roi_mae=roi_edge=roi_stationary=float('nan')
            if k in pair9:
                aa,bb=a[:,:,100:155,125:190],b[:,:,100:155,125:190]; roi_mae=(aa-bb).abs().mean().item(); ax,ay=gradients(aa); bx,by=gradients(bb); roi_edge=((ax-bx).abs().mean()+(ay-by).abs().mean()).item()/2
                if old is not None and old[0]+stride==t and np.linalg.norm(poses[k][t]-poses[k][old[0]])<.001:
                    roi_stationary=((aa-old[1][:,:,100:155,125:190])-(bb-old[2][:,:,100:155,125:190])).abs().mean().item()
            prev.clear(); prev[k]=(t,a.clone(),b.clone())
            rows.append(dict(trajectory=k,frame=t,reference=int(rs[i]),mae=er.mean().item(),mse=er.square().mean().item(),edge_error=((px-gx).abs().mean()+(py-gy).abs().mean()).item()/2,edge_region_error=(region_mean((px-gx).abs(),edge)+region_mean((py-gy).abs(),edge2)).item()/2,temporal_error=temporal,stationary_proxy=stationary,pair9_roi_mae=roi_mae,pair9_roi_edge=roi_edge,pair9_roi_stationary=roi_stationary))
    return rows

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def summarize(rows):
    result={}
    for k in ('mae','mse','edge_error','edge_region_error','temporal_error','stationary_proxy','pair9_roi_mae','pair9_roi_edge','pair9_roi_stationary'):
        v=np.asarray([r[k] for r in rows]); result[k]=float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float('nan')
    return result

@torch.no_grad()
def gif(models,path,keys,device,out):
    with h5py.File(path,'r') as f:
        for k in keys:
            g=f[k]; rgb=g['rgb']; frames=[]
            for t in range(1,len(rgb),2):
                def tensor(j):return convert(torch.from_numpy(rgb[j]).permute(2,0,1)[None],device)
                x=tensor(t); images=[rgb[t]]
                for m in models:
                    r=tensor(1); p,_=m(r,x); images.append((p[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype('uint8'))
                canvas=Image.new('RGB',(256*len(images),286),'white'); draw=ImageDraw.Draw(canvas)
                for i,(im,label) in enumerate(zip(images,['Ground truth']+[m.name for m in models])): canvas.paste(Image.fromarray(im),(256*i,30)); draw.text((256*i+5,5),f'{label} | {t}',fill='black')
                frames.append(canvas)
            frames[0].save(out/f'{k}.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',choices=NAMES,default=NAMES); ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--stride',type=int,default=1); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--run',default='run01'); ap.add_argument('--pair',type=int,default=9); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if min(a.epochs,a.batch,a.stride)<1: raise ValueError('positive epochs/batch/stride required')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=OUT/a.run; out.mkdir(parents=True,exist_ok=False)
    sp=json.loads(SPLIT.read_text()); sets={s:set(map(int,sp[s])) for s in ('train','validation','test')}; assert not any(sets[x]&sets[y] for x,y in [('train','validation'),('train','test'),('validation','test')]);
    with h5py.File(DATA,'r') as f:
        keys={s:[k for k in sorted(f) if k.startswith('trajectory_') and int(f[k].attrs['pair_id']) in ids] for s,ids in sets.items()}; assert sum(map(len,keys.values()))==162; assert all(len(f[k]['rgb'])==223 for v in keys.values() for k in v)
        for v in keys.values():
            for k in v: assert abs(float(f[k]['object_xyz'][2,2]-f[k]['object_xyz'][1,2]))<1e-5
        gifkeys=[k for k in sorted(f) if k.startswith('trajectory_') and int(f[k].attrs['pair_id'])==a.pair]; assert len(gifkeys)==2
        gifsplit=next(s for s,v in sets.items() if a.pair in v)
    if a.smoke: keys={s:v[:2] for s,v in keys.items()}; a.epochs=1; a.stride=100
    (out/'settings.json').write_text(json.dumps(dict(args=vars(a),data=str(DATA),split=sp,reference='original frame 1; frame 0 excluded virtually',device=str(device),gif_pair_split=gifsplit),indent=2))
    print('Device',device,'split sizes',{k:len(v) for k,v in keys.items()},flush=True); summary=[]; models=[]
    for name in a.models:
        seed(a.seed); folder=out/name; folder.mkdir(); m=Candidate(name).to(device); opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-5); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); best=float('inf'); hist=[]
        tr=loader(Frames(DATA,keys['train'],name,a.stride,True),a.batch,a.workers,True); va=loader(Frames(DATA,keys['validation'],name,max(1,a.stride),False),a.batch,a.workers)
        for ep in range(1,a.epochs+1):
            start=time.perf_counter(); tl=run_epoch(m,tr,device,opt,scaler)
            # common validation criterion, not architecture-specific objective
            vr=evaluate(m,DATA,keys['validation'],device,a.batch,a.workers,stride=a.stride); vs=summarize(vr); score=vs['mae']+.25*vs['edge_region_error']+.5*vs['stationary_proxy']; hist.append(dict(epoch=ep,train_loss=tl,score=score,seconds=time.perf_counter()-start,**vs)); write_csv(folder/'history.csv',hist)
            if score<best: best=score; torch.save(dict(model=name,model_state_dict=m.state_dict(),epoch=ep),folder/'best.pt')
            print(name,ep, 'loss',round(tl,6),'val',round(score,6),flush=True)
        m.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True)['model_state_dict']); m.eval()
        tests=[('full',0,0,1),('late50',50,0,1),('late100',100,0,1),('sparse5',0,5,5),('sparse10',0,10,10),('sparse20',0,20,20)]
        if a.smoke: tests=[('smoke',0,0,100)]
        for label,late,gap,stride in tests:
            rows=evaluate(m,DATA,keys['test'],device,a.batch,a.workers,late,gap,stride); write_csv(folder/f'{label}.csv',rows); summary.append(dict(model=name,test=label,parameters=sum(p.numel() for p in m.parameters()),**summarize(rows))); write_csv(out/'summary.csv',summary)
        off=ROOT/'datasets/cleaver_offgrid_48/cleaver_same_goal_162.h5'
        if off.exists() and not a.smoke:
            with h5py.File(off,'r') as f: kk=[k for k in sorted(f) if k.startswith('trajectory_')]
            rows=evaluate(m,off,kk,device,a.batch,a.workers); write_csv(folder/'offgrid.csv',rows); summary.append(dict(model=name,test='offgrid',parameters=sum(p.numel() for p in m.parameters()),**summarize(rows))); write_csv(out/'summary.csv',summary)
        if not a.smoke:
            diag=evaluate(m,DATA,gifkeys,device,a.batch,a.workers); write_csv(folder/'diagnostic_pair.csv',diag); summary.append(dict(model=name,test='diagnostic_pair_'+gifsplit,parameters=sum(p.numel() for p in m.parameters()),**summarize(diag))); write_csv(out/'summary.csv',summary)
            with torch.no_grad():
                x=torch.zeros(1,3,256,256,device=device); m.eval()
                for _ in range(5): m(x,x)
                if device.type=='cuda': torch.cuda.synchronize()
                t0=time.perf_counter()
                for _ in range(20): m(x,x)
                if device.type=='cuda': torch.cuda.synchronize()
                (folder/'timing.json').write_text(json.dumps({'forward_ms':(time.perf_counter()-t0)*50,'device':str(device),'batch':1,'precision':'fp32','includes_reference_encoding':name in REF},indent=2))
        models.append(m.cpu()); del tr,va,opt,scaler; torch.cuda.empty_cache()
    if not a.smoke: gif([m.to(device) for m in models],DATA,gifkeys,device,out)
    print('DONE',out)
if __name__=='__main__': main()
