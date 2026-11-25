"""Frozen encoder; reuse original extraction code with isolated paths."""
import h5py
from CLEAVER.config import ROOT, AE_MODEL
from CLEAVER.core import extract_latents as ex

if __name__=='__main__':
    source=ROOT/'datasets/cleaver_interpolation_162/cleaver_same_goal_162.h5'
    target=ROOT/'datasets/cleaver_interpolation_latent200/cleaver_latent200.h5'
    with h5py.File(source,'r') as h:
        keys={k for k in h if k.startswith('trajectory_')}
        assert keys=={f'trajectory_{i:03d}' for i in range(162)}, 'Incomplete external dataset'
        assert all(h[k]['rgb'].shape==(223,256,256,3) for k in keys)
    ex.RGB_DATA=source
    ex.MODEL_PATH=AE_MODEL
    ex.OUT_DIR=target.parent
    ex.OUT_H5=target
    ex.main()
