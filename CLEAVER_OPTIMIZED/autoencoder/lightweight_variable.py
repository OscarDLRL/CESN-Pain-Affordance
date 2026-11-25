import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, channels, latent=128):
        super().__init__()
        layers=[]
        for c in (16,32,64,128,128):
            layers.extend([nn.Conv2d(channels,c,3,2,1),nn.BatchNorm2d(c),nn.SiLU(inplace=True)])
            channels=c
        self.features=nn.Sequential(*layers)
        self.fc=nn.Linear(128*8*8,latent)
    def forward(self,x):
        return self.fc(self.features(x).flatten(1))

class Decoder(nn.Module):
    def __init__(self, latent, out_channels):
        super().__init__()
        self.fc=nn.Linear(latent,128*8*8)
        layers=[]
        c=128
        # Convolution before interpolation avoids expensive full-resolution feature convolutions.
        for n in (64,32,16,8,8):
            layers.extend([nn.Conv2d(c,n,3,padding=1),nn.BatchNorm2d(n),nn.SiLU(inplace=True),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)])
            c=n
        self.up=nn.Sequential(*layers)
        self.head=nn.Conv2d(c,out_channels,3,padding=1)
    def forward(self,z):
        return self.head(self.up(self.fc(z).reshape(-1,128,8,8)))

class LightweightAutoencoderVariable(nn.Module):
    def __init__(self, dynamic_dim=128):
        super().__init__()
        self.static_latent_dim=128
        self.dynamic_latent_dim=dynamic_dim
        self.static_encoder=Encoder(3)
        self.dynamic_encoder=Encoder(5,dynamic_dim)
        self.static_aux_decoder=Decoder(128,3)
        self.fuse=nn.Sequential(nn.Linear(128+dynamic_dim,256),nn.SiLU(inplace=True))
        self.dynamic_decoder=Decoder(256,4)
        self.motion_centroid_head=nn.Sequential(nn.Linear(dynamic_dim,64),nn.SiLU(),nn.Linear(64,2),nn.Tanh())
        yy,xx=torch.meshgrid(torch.linspace(-1,1,256),torch.linspace(-1,1,256),indexing='ij')
        self.register_buffer('coordinates',torch.stack((xx,yy))[None],persistent=False)
    def encode_static(self,x): return self.static_encoder(x)
    def encode_dynamic(self,delta):
        return self.dynamic_encoder(torch.cat((delta,self.coordinates.to(delta.dtype).expand(len(delta),-1,-1,-1)),1))
    def decode_static_aux(self,z): return self.static_aux_decoder(z).sigmoid()
    def decode_dynamic_full(self,z_dynamic,z_static):
        h=self.dynamic_decoder(self.fuse(torch.cat((z_dynamic,z_static),1)))
        candidate=h[:,:3].tanh(); logits=h[:,3:]; mask=logits.sigmoid()
        return dict(candidate_delta=candidate,motion_mask_logits=logits,motion_mask=mask,delta_recon=candidate*mask)
    def forward(self,frame0,frame_t,cached_z_static=None,cached_static_mse=None):
        delta=frame_t-frame0
        zs=self.encode_static(frame0) if cached_z_static is None else cached_z_static
        zd=self.encode_dynamic(delta)
        out=self.decode_dynamic_full(zd,zs)
        return dict(out,z_static=zs,z_dynamic=zd,delta=delta,static_aux_recon=self.decode_static_aux(zs) if cached_static_mse is None else None,cached_static_mse=cached_static_mse,frame_recon=(frame0+out['delta_recon']).clamp(0,1),motion_centroid=self.motion_centroid_head(zd))
