import os
import glob
import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors
from omegaconf import DictConfig
import shutil

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO

import os
import tqdm
import numpy as np
import torch
import trimesh
import os, sys 
from lvd_templ.data.datamodule_AMASS import MetaData
from lvd_templ.paths import path_challenge_pairs
from lvd_templ.paths import  chk_pts, home_dir, output_dir

os.chdir(home_dir)

sys.path.append("./data/preprocess_voxels/libvoxelize")
sys.path.append("./data/preprocess_voxels/")
sys.path.append("./data/")
sys.path.append(".")
sys.path.append('./src')
sys.path.append('./preprocess_voxels')
sys.path.append('/home/ubutnu/Documents/Projects/LVD_templ/lvd_templ/preprocess_voxels')
sys.path.append('/home/ubutnu/Documents/Projects/LVD_templ/lvd_templ/preprocess_voxels/libvoxelize')

device = torch.device('cuda')


def get_name_parser(challenge):
    if challenge.name=='FAUST_train_scans':
        return lambda x: 'tr_scan_' + x.zfill(3)
    if challenge.name=='FAUST_train_reg':
        return lambda x: 'tr_reg_' + x.zfill(3)

def get_match_reg(s_src, s_tar, reg_src, reg_tar):
    # Returns for each point of s_src the match for s_tar
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(reg_src)
    distances, result_s = nbrs.kneighbors(s_src)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(s_tar)
    distances, result_t = nbrs.kneighbors(reg_tar[np.squeeze(result_s)])

    result = np.squeeze(result_t)
    return result


def get_dataset(challenge):
    print(challenge)
    if challenge.name=='FAUST_train_scans':
        return path_challenge_pairs, '', '_'
    if challenge.name=='FAUST_train_reg':
        return path_challenge_pairs, '', '_'
    raise ValueError('this challenge does not exists')


def get_model(chk):
    chk_zip = glob.glob(chk + 'checkpoints/*.zip')[0]
    ### Restore the network
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../../../" + str(chk))
    cfg_model = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg_model))
    train_data = hydra.utils.instantiate(cfg_model.nn.data.datasets.train, mode="test")
    MD = MetaData(class_vocab=train_data.class_vocab)
    model: pl.LightningModule = hydra.utils.instantiate(cfg_model.nn.module, _recursive_=False, metadata=MD)
    old_checkpoint = NNCheckpointIO.load(path=chk_zip)

    module = model._load_model_state(checkpoint=old_checkpoint, metadata=MD).to(device)
    module.model.eval()
    
    return module, MD, train_data, cfg_model


def run(cfg: DictConfig) -> str:
    os.chdir(home_dir)
    
    DATA_DIR = cfg['core'].datadir
    model_name = cfg['core'].checkpoint
    pairs, tag, split_symb = get_dataset(cfg['core'].challenge) 
    target = cfg['core'].target
    regist = cfg['core'].regist
    
    out_dir = DATA_DIR  + cfg['core'].challenge.name + '/_' + model_name + '_' + regist + '_' + str(cfg['core'].subdiv) + '_' + tag +'/'  
    
    if not(os.path.exists(DATA_DIR  + cfg['core'].challenge.name  )):
         os.mkdir(DATA_DIR + cfg['core'].challenge.name )
        
    
    print(DATA_DIR + model_name + '/' + cfg['core'].challenge.name)     
    if not(os.path.exists(DATA_DIR + model_name + '/' + cfg['core'].challenge.name )):
        raise ValueError('Registrations not available')
    
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)
    else:
        if cfg['core'].overwrite:
            os.rmdir(out_dir)
            os.mkdir(out_dir)
        else:
            exit 
    
    file1 = open(pairs, 'r')
    Lines = file1.read().splitlines()
    x = [Lines[i].split(split_symb) for i in range(len(Lines))]
   
    base_path = DATA_DIR + model_name + '/' + cfg['core'].challenge.name
    
    parser = get_name_parser(cfg['core'].challenge)
    
    
    
    ### Computing matches
    for pair in tqdm.tqdm(x):
        
        # Target Shapes
        A_mesh = base_path  + '/' + parser(pair[0]) + '/' + target + '.ply'
        B_mesh = base_path  + '/' + parser(pair[1])  + '/' + target + '.ply'
        
        A = trimesh.load(A_mesh, process=False, maintain_order=True)
        B = trimesh.load(B_mesh, process=False, maintain_order=True)
        
        # Registrations
        A_reg = base_path  + '/' + parser(pair[0]) + '/' + regist + '.ply'
        B_reg = base_path  + '/' + parser(pair[1])  + '/' + regist + '.ply'
        A_our = trimesh.load(A_reg, process=False, maintain_order=True)
        B_our = trimesh.load(B_reg, process=False, maintain_order=True)    
        
        # If a denser match is required, use template subdivision
        if cfg['core'].subdiv:
            for _ in np.arange(cfg['core'].subdiv):
                A_our = A_our.subdivide()
                B_our = B_our.subdivide()
        
        m = get_match_reg(np.asarray(A.vertices), np.asarray(B.vertices), np.asarray(A_our.vertices), np.asarray(B_our.vertices))
        
        # Write the correspondence
        with open(out_dir + pair[0] + '_' + pair[1] + '.txt', 'w') as f:
            f.write("\n".join(map(str, m)))
            f.write("\n")
    
@hydra.main(config_path=str(PROJECT_ROOT / "conf_test"), config_name="default_match")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
    

