from __future__ import division

import os
import sys
from itertools import combinations, permutations

import hydra
import omegaconf
import torch
from torch.utils.data import Dataset
from pathlib import Path
PROJECT_ROOT = Path.cwd()
from nn_core.nn_types import Split

################

import os
from os import path as osp

import random
import hydra
import numpy as np
import omegaconf
import torch
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import Dataset
import open3d as o3d
from sklearn.neighbors import KDTree
import robust_laplacian
import scipy.sparse.linalg as sla
from sklearn.cluster import KMeans
import scipy.io as sio 

################

# PATH TO NEUTRAL SMPL MODEL
bm_fname = "./support_data/body_models/smplh/neutral/model.npz"

num_betas = 16  # number of body parameters
num_dmpls = 8   # number of DMPL parameters


class AMASSDataset(Dataset):
    def __init__(
        self,
        mode,
        num_betas=16, 
        batch_size=64,
        num_workers=12,
        **kwargs
    ):
        # TRAIN, VALIDATION, TEST?
        self.mode = mode 
        
        # What device we use WARNING: NOT YET CHECKED IF IT IS ACTUALLY OK TO CHANGE IT
        self.comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set DATAPATH
        self.data_path = kwargs["data_path"]
        self.path = self.data_path + kwargs["version"] + "/stage_III/"
        
        # Values from KWARGS
        self.n_points = kwargs["n_points"]
        self.factor = kwargs["red_factor"]
        self.type = kwargs["type"]
        self.segm = kwargs["segm"]
        random.seed(kwargs["seed_idxs"])
        
        # Voxalization resolution
        self.occ_res  = kwargs["res"]
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.support_dir = "./support_data/"
        # How many betas?
        self.num_betas = num_betas
        
        # LVD TRAINING SAMPLING STRATEGY
        self.fine_std  =  kwargs["fine_std"]
        self.n_uniform =  kwargs["n_uniform"]
        self.n_fine_sampled = kwargs["n_fine_sampled"]   
                    
        # Template Shape
        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas) 
        self.template = bm.forward(root_orient=torch.tensor([[np.pi / 2, 0, 0]])).v.to(self.comp_device ) 
        
        # If we want a multi-head LVD, we do surface clustering on the template to specialize each head
        if self.segm>0:
            template = np.squeeze(np.asarray(bm.forward().v))
            template_f = np.asarray(bm.f)
            L, A = robust_laplacian.mesh_laplacian(template, template_f)            # SMPL Laplacian
            evals, evecs = sla.eigsh(L, self.segm, A, sigma=1e-8)                   # Laplacian Eigendecomposition
            kmeans = KMeans(n_clusters=self.segm, random_state=0).fit(evecs)        # K-Means on eigenvectors
            self.labels = kmeans.labels_                                            # SAVE LABELS
            np.save('segms_' + str(self.segm), self.labels)                         # Save it locally to guarantee reproducibility
        else:
            self.labels = []
            
        # HOW MANY POINTS WE WANT TO MATCH FROM THE TEMPLATE?    
        p = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.squeeze(self.template.cpu().numpy())),
                                      o3d.utility.Vector3iVector(bm.f))
        
        # If not set -> keep original resolution
        if self.factor == 1 and self.n_points==0:
            self.idxs = np.arange(0,6890)       
        
        # If factor is not set but n_points yes, use random n_points                         
        elif self.factor == 1:
            self.idxs = random.sample(range(0, 6890), self.n_points)            #TODO: proper SAMPLING 
        else:
            
            # If factor is set, reduce the template vertices using the specified factor
            p = p.simplify_quadric_decimation(bm.f.shape[0]//self.factor)                           # meaningful sampling
            kdt = KDTree(np.squeeze(self.template.cpu().numpy()), metric='euclidean')               # Find indexs on the template
            self.idxs = np.squeeze(kdt.query(np.asarray(p.vertices), k=1, return_distance=False))   # Store the idxs
            np.save('idxs_' + str(self.factor), self.idxs)                                          # Save it locally to guarantee reproducibility
            self.red_templ = self.template[:,self.idxs]                                             # Save our reduced template
            
        self.faces = np.asarray(p.triangles)                                                        #Number of faces
        self.if_templ = kwargs["template"]
        self.ds = {}
        # We take the length from betas
        print(os.path.join(self.path, str(mode), "betas.pt"))
        b = torch.load(os.path.join(self.path, str(mode), "betas.pt"))                            
        self.len = b.shape[0] if kwargs["n_data"] == 0 else np.min((kwargs["n_data"],b.shape[0]))
        del b
        
        # This is to use Fourier coefficients instead of voxels
        self.coeffs = kwargs["coeffs"]
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = {}
        # Take the vertices
        data["data_v_scal"] = torch.load(os.path.join(self.path, str(self.mode), "ifnet_indi", "verts_" + str(self.type) , str(f'{int(idx):09}')) + ".pt")
        
        # Load the voxelization
        if self.coeffs:
            data["occ"] = torch.load(os.path.join(self.path, str(self.mode), "ifnet_indi", "coeff_occ_dist", str(f'{int(idx):09}')) + ".pt")
            data["occ"] = torch.reshape((self.laplacian @ data["occ"]),(self.occ_res,self.occ_res,self.occ_res))
        else:
            data["occ"] = torch.load(os.path.join(self.path, str(self.mode), "ifnet_indi", self.type, str(f'{int(idx):09}')) + ".pt")
        
        if self.type == 'occ':
            data["occ"] = np.unpackbits(data["occ"]).astype(np.uint8)
        

        # Sample random points in the voxelization sapce
        b_min = np.array([-0.8, -0.8, -0.8])
        b_max = np.array([ 0.8,  0.8,  0.8])
        rand_uniform = np.random.uniform(b_min, b_max, (self.n_uniform, 3))
        
        # Sample random points near the SMPL surface
        smpl_inds = np.arange(6890)
        np.random.shuffle(smpl_inds)
        smpl_inds = smpl_inds[:self.n_fine_sampled]
        noise_smpl = np.random.normal(0, self.fine_std, (self.n_fine_sampled, 3))
        rand_smpl =  data["data_v_scal"][smpl_inds] + noise_smpl
        point_pos = np.concatenate((rand_uniform, rand_smpl))
        _target_smpl = data["data_v_scal"][self.idxs]


        local_smpl = 0
        smpl_inds_wgt = 0

        sample = {
                  'input_voxels':  torch.as_tensor(data["occ"][None]),
                  'input_points':  torch.as_tensor(point_pos,dtype=torch.float32),
                  'smpl_vertices': torch.as_tensor(_target_smpl,dtype=torch.float32),
                  'idxs': self.idxs,
                  'smpl_vertices_full': torch.as_tensor(data["data_v_scal"],dtype=torch.float32),
                  'smpl_inds': smpl_inds,
                  'smpl_inds_wgt': smpl_inds_wgt,
                  'local_smpl': torch.as_tensor(local_smpl,dtype=torch.float32),
                  'labels': self.labels
                  }

        return sample


    def class_vocab(self):
        dd = {}
        dd['occ_res']   = self.occ_res
        dd['gt_points'] = len(self.idxs)
        dd['labels'] = self.labels
        dd['idxs'] = self.idxs
        return dd

    def get_loader(self, shuffle=False):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
        )

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def new_idxs(self):
        self.idxs = random.sample(range(0, 6890), self.n_points)
        return

@hydra.main(config_path=str(PROJECT_ROOT / "conf_ifnet"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    # pippo= hydra.utils.instantiate(
    #    cfg.nn.data.datasets.train,root_dir= PROJECT_ROOT, name="faust", train=True, use_cache=True, op_cache_dir=None
    # )
    test = hydra.utils.instantiate(cfg.nn.data.datasets.train, mode="train")

    print(len(test))
    print(str(test[0]))


if __name__ == "__main__":
    main()
