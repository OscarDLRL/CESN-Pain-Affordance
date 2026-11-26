"""Final static-preservation ablation; uses V6-SE with frozen encoder."""
import argparse, itertools, hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from CLEAVER_OPTIMIZED.experiments import train_v6_compact128 as v
from CLEAVER_OPTIMIZED.experiments.run_v6_shared_encoder import SharedV6
from CLEAVER_OPTIMIZED.experiments.run_v6_residual_ablation import reconstruct, measures, csvsave

# Geometry and camera from FINAL162's v4 generator / paired reaches environment.
def corners(center, half):
    return np.array(center)+np.array(list(itertools.product([-1,1],repeat=3)))*half
BOXES=[corners([0,0,0],[.09,.0015,.045]),corners([.15,0,-.027],[.06,.012,.014])]
EYE=np.array([.60,-.70,.60]); TARGET=np.array([0,0,.12]); FORWARD=TARGET-EYE; FORWARD/=np.linalg.norm(FORWARD)
RIGHT=np.cross(FORWARD,[0,0,1]); RIGHT/=np.linalg.norm(RIGHT); UP=np.cross(RIGHT,FORWARD)
FOCAL=128/np.tan(.5)
def projected_mask(pos,q):
    rot=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]) # SAPIEN wxyz
    im=Image.new('L',(256,256)); draw=ImageDraw.Draw(im)
    for box in BOXES:
        p=rot.apply(box)+pos-EYE; depth=p@FORWARD
        if np.any(depth<=0): raise ValueError('Object behind camera')
        uv=np.stack([128+FOCAL*(p@RIGHT)/depth,128-FOCAL*(p@UP)/depth],1)
        hull=ConvexHull(uv); draw.polygon([tuple(x) for x in uv[hull.vertices]],fill=1)
    return np.asarray(im,dtype=np.uint8)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--diagnose',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--stride',type=int,default=2); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weights',type=float,nargs='+',default=[0,2,8]); p.add_argument('--out',type=Path); a=p.parse_args()
    assert min(a.epochs,a.batch_size,a.stride)>0 and all(w>=0 for w in a.weights)
    root=v.cfg.OUT; src=root/'v6_shared_encoder'/'V6-SE.pt'; out=a.out or root/('v6_preserve_diagnostic' if a.diagnose else 'v6_preserve_smoke' if a.smoke else 'v6_static_preservation'); out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(src,map_location='cpu',weights_only=False); assert ck['encoder_mode']=='V6-SE' and ck['reconstruction']=='alpha'
    keys,pair,labels,obj,rgb=v.load_data(); tr,va,te=v.split_pairs(pair,obj)
    for k,ids in [('final_train_pairs',tr),('validation_pairs',va),('test_pairs',te)]: assert list(ck[k])==list(ids)
    train,val=[np.flatnonzero(np.isin(pair,x)) for x in (tr,va)]; assert len(train)==108 and len(val)==18
    # Store compact ROI bounding rectangles around projected silhouettes, not full image masks.
    boxes=np.zeros((len(keys),223,4),dtype=np.int16)
    with h5py.File(v.cfg.RGB,'r') as f:
        print('Dataset keys:',list(f[keys[0]].keys()),flush=True)
        for i in np.r_[train,val]:
            g=f[keys[i]]; pos=g['object_xyz'][:]; q=g['object_quat'][:]; assert pos.shape==(223,3) and q.shape==(223,4)
            for t in range(223):
                mask=projected_mask(pos[t],q[t]); yy,xx=np.where(mask)
                if not len(xx): raise ValueError(f'Empty projected ROI {keys[i]} {t}')
                boxes[i,t]=[max(0,xx.min()-4),max(0,yy.min()-4),min(256,xx.max()+5),min(256,yy.max()+5)]
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    frames=[0,25,75,125,175,222]; examples=[]
    for group in (train,val):
        for oid in np.unique(obj[group]):
            ii=group[obj[group]==oid]; pid=pair[ii[0]]
            examples.extend(int(np.flatnonzero((pair==pid)&(labels==l))[0]) for l in (0,1))
    def roi(i,t,union=False):
        b=torch.as_tensor(boxes[i,t],device='cuda'); xx=torch.arange(256,device='cuda')[None,None,:]; yy=torch.arange(256,device='cuda')[None,:,None]
        def make(b): return ((xx>=b[:,0,None,None])&(xx<b[:,2,None,None])&(yy>=b[:,1,None,None])&(yy<b[:,3,None,None]))[:,None]
        m=make(b)
        return m|make(torch.as_tensor(boxes[i,0],device='cuda')) if union else m
    def mean_region(err,m): return ((err.float()*m).sum((1,2,3))/(3*m.sum((1,2,3)).clamp_min(1))).mean()
    def batches(ids,stride,epoch=None):
        ii,tt=np.meshgrid(ids,np.arange(0,223,stride),indexing='ij'); ii=ii.ravel(); tt=tt.ravel(); order=np.arange(len(ii))
        if epoch is not None: np.random.default_rng(42+epoch).shuffle(order)
        if a.smoke: order=order[:2*a.batch_size]
        for j in range(0,len(order),a.batch_size):
            k=order[j:j+a.batch_size]; yield ii[k],tt[k]
    def make_model():
        m=SharedV6('V6-SE').cuda(); m.load_state_dict(ck['model_state_dict'],strict=True); m.requires_grad_(False).eval(); return m
    base=make_model(); s,mi=v.cache_static(base,rgb,np.r_[train,val]); z=torch.empty((len(keys),223,128),device='cuda',dtype=torch.float16)
    with torch.no_grad(),v.amp():
        for i in np.r_[train,val]:
            x=v.device_rgb(rgb[i,0:1]);
            for t in range(0,223,a.batch_size):
                y=v.device_rgb(rgb[i,t:t+a.batch_size]); z[i,t:t+len(y)]=base.encode_dynamic(y-x)
    def run(m,w,epoch=None,opt=None,scaler=None):
        training=opt is not None; m.eval(); sums={}; n=0
        for i,t in batches(train if training else val,a.stride if training else 1,epoch):
            x=v.device_rgb(rgb[i,0]); y=v.device_rgb(rgb[i,t]); r=roi(i,t); u=roi(i,t,True)
            if training: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training),v.amp():
                o=reconstruct(m,x,z[i,t],s[i],mi[i],True); o['delta']=y-x; pred=o['frame_recon']; met=measures(o,x,y)
                objloss=mean_region((pred-y).square(),u); edge=mean_region((pred[:,:,:,1:]-pred[:,:,:,:-1]-y[:,:,:,1:]+y[:,:,:,:-1]).abs(),r[:,:,:,1:]|r[:,:,:,:-1])+mean_region((pred[:,:,1:,:]-pred[:,:,:-1,:]-y[:,:,1:,:]+y[:,:,:-1,:]).abs(),r[:,:,1:,:]|r[:,:,:-1,:]); stable=(y-x).abs().amax(1,keepdim=True)<=.02
                # Equal weighting of global stable pixels and stable pixels near the object.
                preserve=.5*mean_region((pred-x).square(),stable)+.5*mean_region((pred-x).square(),stable&u)
                falsemask=.5*mean_region(o['motion_mask'].square().expand(-1,3,-1,-1),stable)+.5*mean_region(o['motion_mask'].square().expand(-1,3,-1,-1),stable&u)
                loss=v.full_losses(o,x,y)['total']+4*met['region_loss']+w*(preserve+falsemask)
            if training: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            for k,b in dict(met,object_roi_mse=mean_region((pred-y).square(),r),object_union_mse=objloss,object_edge_l1=edge,stable_preserve_mse=preserve,stable_mask_penalty=falsemask,objective=loss).items(): sums[k]=sums.get(k,0.)+float(b.detach())*len(i)
            n+=len(i)
        return {k:x/n for k,x in sums.items()}
    def cycle(m):
        sums=np.zeros(4); n=0
        with torch.no_grad(),v.amp():
            for i,t in batches(val,10):
                x=v.device_rgb(rgb[i,0]); y=v.device_rgb(rgb[i,t]); zz=z[i,t]; cur=zz; r=roi(i,t)
                for k in range(5):
                    pred=reconstruct(m,x,cur,s[i],None,True)['frame_recon']; cur=m.encode_dynamic(pred-x)
                    if k in (0,4):
                        off=0 if k==0 else 2; sums[off]+=float(((cur.float()-zz.float()).norm(dim=1)/zz.float().norm(dim=1).clamp_min(1e-6)).mean())*len(i); sums[off+1]+=float(mean_region((pred-y).square(),r))*len(i)
                n+=len(i)
        return dict(zip(['cycle1_relative_l2','cycle1_object_roi_mse','cycle5_relative_l2','cycle5_object_roi_mse'],sums/n))
    hist=[]; summary=[]; images={}; frames=[0,25,75,125,175,222]; pid=pair[val[0]]; examples=[int(np.flatnonzero((pair==pid)&(labels==l))[0]) for l in (0,1)]
    def capture(name,m):
        images[name]=[]
        with torch.no_grad(),v.amp():
            for i in examples:
                x=v.device_rgb(rgb[[i]*len(frames),0]); pred=reconstruct(m,x,z[i,frames],s[[i]*len(frames)],None,True)['frame_recon']; images[name].append(pred.cpu().numpy().transpose(0,2,3,1))
    def temporal(name,m):
        rows=[]
        with torch.no_grad(),v.amp():
            for t in [0,1,5,10,25,50,75,125,175,222]:
                i=val; ts=np.full(len(i),t); x=v.device_rgb(rgb[i,0]); y=v.device_rgb(rgb[i,t]); r=roi(i,ts,True)
                o=reconstruct(m,x,z[i,t],s[i],None,True); pred=o['frame_recon']; stable=(y-x).abs().amax(1,keepdim=True)<=.02; mask=o['motion_mask']; q=r&stable
                def masked(a,b): return float((a.float()*b).sum()/b.expand_as(a).sum().clamp_min(1))
                rows.append(dict(model=name,frame=t,actual_roi_change=masked((y-x).abs(),r),stable_roi_fraction=float(q.sum()/r.sum().clamp_min(1)),stable_roi_mask_mean=masked(mask,q),stable_roi_false_motion_rate=masked((mask>.5).float(),q),stable_roi_preserve_mse=masked((pred-x).square(),q),object_roi_mse=masked((pred-y).square(),r),rgb_mse=float((pred-y).square().mean())))
        return rows
    diagnostic=temporal('V6-SE-base',base); csvsave(out/'temporal_diagnostic.csv',diagnostic)
    if a.diagnose:
        for row in diagnostic: print(row,flush=True)
        print('DIAGNOSTIC COMPLETE',out,flush=True); return
    baseline=run(base,0); summary.append(dict(model='V6-SE-base',epoch=0,**baseline,**cycle(base))); capture('V6-SE-base',base)
    for w in a.weights:
        name=f'V6-SE-P{w:g}'; v.seed_everything(42); m=make_model(); m.dynamic_decoder.requires_grad_(True); opt=torch.optim.AdamW(m.dynamic_decoder.parameters(),lr=a.lr,weight_decay=1e-5); scaler=torch.amp.GradScaler('cuda'); best=float('inf'); chosen=None
        for ep in range(1,(1 if a.smoke else a.epochs)+1):
            run(m,w,ep,opt,scaler); met=run(m,w); row=dict(model=name,epoch=ep,**met); hist.append(row); print(row,flush=True)
            # Same selection criterion across all weights; limit moving-region degradation to 5%.
            score=met['object_union_mse'] if met['moving_mse']<=1.05*baseline['moving_mse'] else 1+met['object_union_mse']
            if score<best:
                best=score; chosen=row; v.atomic_save(dict(ck,model_state_dict=m.state_dict(),preservation_weight=w,preservation_epoch=ep),out/(name+'.pt'))
            csvsave(out/'history.csv',hist)
        m.load_state_dict(torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)['model_state_dict']); summary.append(dict(chosen,**cycle(m))); capture(name,m); diagnostic.extend(temporal(name,m)); csvsave(out/'temporal_diagnostic.csv',diagnostic); csvsave(out/'comparison.csv',summary)
    for l,label in enumerate(('safe','pain')):
        panels={'Original':rgb[examples[l],frames]/255.,**{k:x[l] for k,x in images.items()}}; fig,ax=plt.subplots(len(panels),len(frames),figsize=(13,2.1*len(panels)))
        for r,(name,im) in enumerate(panels.items()):
            for c,t in enumerate(frames):
                ax[r,c].imshow(im[c].clip(0,1)); ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
                if c==0: ax[r,c].set_ylabel(name)
                if r==0: ax[r,c].set_title(f'Frame {t}')
        fig.tight_layout(); fig.savefig(out/f'comparison_{label}.png',dpi=300); plt.close(fig)
    v.write_json(out/'settings.json',dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),roi='projected blade and handle bounding rectangle + 4 px; union initial/current for loss',camera_eye=EYE.tolist(),camera_target=TARGET.tolist(),fov=1.,arguments={k:str(x) if isinstance(x,Path) else x for k,x in vars(a).items()},selection='minimum validation object union MSE with preference for moving MSE <= 105% of baseline',stable_threshold=.02,preservation='equal global and object-union stable RGB MSE plus squared soft mask activation',train_pairs=tr,validation_pairs=va,test_pairs=te)); print('COMPLETE',out,flush=True)
if __name__=='__main__': main()
