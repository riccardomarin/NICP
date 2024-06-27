import os
import sys
import logging


from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger
from lvd_templ.data.datamodule_AMASS import MetaData
from lvd_templ.modules.module import Network_LVD, Network_LoVD

from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

import scipy.io as sio 

pylogger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

s_max = torch.nn.Softmax(dim=1)

num_betas = 16  # number of body parameters
num_dmpls = 8   # number of DMPL parameters

######## UTILITY FUNCTIONS #############
def plot_RGBmap(
    verts,
    trivs,
    colors=None,
    colorscale=[[0, "rgb(0,0,255)"], [0.5, "rgb(255,255,255)"], [1, "rgb(255,0,0)"]],
    point_size=3,
):
    "Draw multiple triangle meshes side by side"
    "colors must be list(range(num colors))"
    if type(verts) is not list:
        verts = [verts]
    if type(trivs) is not list:
        trivs = [trivs]
    if type(colors) is not list:
        colors = [colors]
    if type(verts[0]) == torch.Tensor:
        to_np = lambda x: x.detach().cpu().numpy()
        verts = [to_np(v) for v in verts]

    "Check device for torch tensors"

    def to_cpu(v):
        if torch.is_tensor(v):
            return v.data.cpu()
        return v

    verts = [to_cpu(x) for x in verts]
    trivs = [to_cpu(x) for x in trivs]
    colors = [to_cpu(x) for x in colors]

    nshapes = min([len(verts), len(colors), len(trivs)])

    fig = make_subplots(rows=1, cols=nshapes, specs=[[{"type": "surface"} for i in range(nshapes)]])

    for i, [vert, triv, col] in enumerate(zip(verts, trivs, colors)):
        if triv is not None:
            if col is not None:
                mesh = go.Mesh3d(
                    x=vert[:, 0],
                    z=vert[:, 1],
                    y=vert[:, 2],
                    i=triv[:, 0],
                    j=triv[:, 1],
                    k=triv[:, 2],
                    vertexcolor=col,
                    colorscale=colorscale,
                    color="lightpink",
                    opacity=1,
                )
            else:
                mesh = go.Mesh3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2], i=triv[:, 0], j=triv[:, 1], k=triv[:, 2])
        else:
            if col is not None:
                mesh = go.Scatter3d(
                    x=vert[:, 0],
                    z=vert[:, 1],
                    y=vert[:, 2],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        color=col,  # set color to an array/list of desired values
                        # choose a colorscale
                        opacity=1,
                    ),
                )
            else:
                mesh = go.Scatter3d(
                    x=vert[:, 0],
                    z=vert[:, 1],
                    y=vert[:, 2],
                    mode="markers",
                    marker=dict(
                        size=point_size,  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=1,
                    ),
                )

        fig.add_trace(mesh, row=1, col=i + 1)
        fig.get_subplot(1, i + 1).aspectmode = "data"

        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=4, z=-1))
        fig.get_subplot(1, i + 1).camera = camera

    #     fig = go.Figure(data=[mesh], layout=layout)
    fig.update_layout(
        #       autosize=True,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    # paper_bgcolor="LightSteelBlue")
    # fig.show()
    return fig


def pcs_renderings(src, tar, m):
    c_src = color_f(src)
    c_tar = torch.cat([torch.unsqueeze(c_src[k][m[k]], 0) for k in range(src.shape[0])])

    src = torch.dstack([src, c_src]).detach().cpu().numpy()
    tar = torch.dstack([tar + torch.tensor([1, 0, 0]).to(tar.device), c_tar]).detach().cpu().numpy()
    both = np.concatenate([src, tar], 1)

    return src, tar, both

def color_f(src, freq=1):
    funz_ = torch.dstack(
        [
            (src[k, :, :] - torch.min(src[k, :, :], 0)[0])
            / torch.tile((torch.max(src[k, :, :], 0)[0] - torch.min(src[k, :, :], 0)[0]), (src[k, :, :].shape[0], 1))
            for k in range(src.shape[0])
        ]
    )
    funz_ = torch.transpose(torch.transpose(funz_, 0, 2), 1, 2)
    colors = torch.cos(freq * funz_) * 255
    return colors  # (colors-np.min(colors))/(np.max(colors) - np.min(colors))


def render_result(res):
    fig = plot_RGBmap([res[:, 0:3]], [None], colors=[res[:, 3:]])
    return fig
####### END UTILITY FUNCTIONS ############

####### START LOSS FUNCTIONS ############
def loss_uni(basis_A, basis_B, pc_B):

    # SoftMap
    dist_matrix = torch.cdist(basis_A, basis_B)
    s_max_matrix = s_max(-dist_matrix)

    # Basis Loss
    loss = torch.sum(torch.square(torch.matmul(s_max_matrix, pc_B) - pc_B)) / basis_A.shape[0]
    return loss

def loss_basis(basis_A, basis_B, pc_B):
        # Computing optimal transformation
        pseudo_inv_A = torch.pinverse(basis_A.cpu()).cuda()
        C_opt = torch.matmul(pseudo_inv_A, basis_B)
        opt_A = torch.matmul(basis_A, C_opt)
        # SoftMap
        #dist_matrix =        
        
        pc_B_t = s_max(-torch.cdist(opt_A,basis_B))@pc_B

        # Basis Loss
        loss = torch.sum(torch.square(pc_B_t - pc_B))
        #loss = torch.sum(torch.square(s_max_matrix - torch.eye(s_max_matrix.shape[1]).unsqueeze(0).cuda()))
        return loss

def loss_desc(phi_A,phi_B,G_A,G_B):
        # Computing optimal transformation
        p_inv_phi_A = torch.pinverse(phi_A.cpu()).cuda()
        p_inv_phi_B = torch.pinverse(phi_B.cpu()).cuda()
        c_G_A = torch.matmul(p_inv_phi_A, G_A)
        c_G_B = torch.matmul(p_inv_phi_B, G_B)
        c_G_At = torch.transpose(c_G_A,2,1)
        c_G_Bt = torch.transpose(c_G_B,2,1)

        # Estimated C
        #C_my = torch.matmul(c_G_A,torch.transpose(torch.pinverse(c_G_Bt),2,1))
        C_my = c_G_A @ torch.transpose(torch.pinverse(c_G_Bt.cpu()).cuda(),2,1)
        # Optimal C
        C_opt = torch.matmul(p_inv_phi_A, phi_B)

        # MSE
        eucl_loss = torch.mean(torch.square(C_opt - C_my))

        return eucl_loss
####### END LOSS FUNCTIONS ############


class LightUniversal(pl.LightningModule):
    logger: NNLogger
                
    ### THIS IS A FUNCTION TO FETCH VALUES FROM THE CONFIG   
    def fetch_kwargs(self, kwargs):
        
        
        # USe of external device
        if "gpus" in kwargs.keys():
            self.gpus = kwargs["gpus"]
            self.dev = torch.device("cuda" if self.gpus else "cpu")
        else:
            self.gpus = 1
            self.dev = torch.device("cuda" if self.gpus else "cpu")

        # Do we want to match against a template?
        self.if_templ = kwargs["template"]
        if self.if_templ:
            bm = BodyModel(bm_fname=kwargs['smpl_path'], num_betas=num_betas)
            self.template = bm.forward().v.to(self.dev)
        
        # N Points    
        self.n_points = kwargs["n_points"]
         
        # Set the clamp for the LVD Prediction
        if "clamp" in kwargs.keys():
            self.clamp = kwargs["clamp"]
        else:
            self.clamp = 0.3
           
        # SELFSUP Training -> Does not work
        if "selfsup" in kwargs.keys():
            self.selfsup = kwargs["selfsup"]
        else:
            self.selfsup = False
        
        # Do we want to use a powerful IFNET?
        if "powerup" in kwargs.keys():
            self.powerup = kwargs["powerup"]
        else:
            self.powerup = False
            
        # How much powerful?   
        if "power_factor" in kwargs.keys():
            self.power_factor = kwargs["power_factor"]
        else:
            self.power_factor = 1

        # Layers size of the LVD heads
        if "size_layers" in kwargs.keys():
            self.size_layers = kwargs["size_layers"]
        else:
            self.size_layers = 2048
                        
        # Do you want to use SDF gradients as features?
        if "grad" in kwargs.keys():
            if kwargs["grad"] > 0:
                self.grad = kwargs["grad"]
                self.input_dim = 7 + 6
            else:
                self.grad = 0
                self.input_dim = 7                
        else:
            self.grad = 0
            self.input_dim = 7
        
        # GENERAL LVD PARAMETERS FOR TRAINING
        if "fine_std" in kwargs.keys():    
            self.fine_std  =  kwargs["fine_std"]
            self.n_uniform =  kwargs["n_uniform"]
            self.n_fine_sampled = kwargs["n_fine_sampled"]
        else:
            # Random points in the space
            self.n_uniform =  400
            # Points subsampled near SMPL surface
            self.n_fine_sampled = 1800
            self.fine_std  =  0.05     
        

        # clamp_style=0 is basically wrong
        if "clamp_style" in kwargs.keys():
            self.clamp_style = kwargs["clamp_style"]
            self.max_norm = np.linalg.norm(np.asarray([self.clamp, self.clamp, self.clamp]),2)
        else:
            self.clamp_style = 0
        
        # Number of LVD heads
        if "segm" in kwargs.keys():
            self.segm = kwargs["segm"]
        else:
            self.segm = 0
        
                           
        if "paradigm" in kwargs.keys():
            self.paradigm = kwargs["paradigm"]
        else:
            self.behave = "LoVD"                
                
        
    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Storing run hyperparams
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.metadata = metadata
        
        self.count_steps = 0
        
        # If metadata is not passed: set max  occ_res and gt_points
        if metadata is None:
            self.occ_res = 128
            self.gt_points = 6890
        else:
            self.occ_res = metadata.class_vocab()['occ_res']
            self.gt_points = metadata.class_vocab()['gt_points']
            self.idxs = metadata.class_vocab()['idxs']
            if len(metadata.class_vocab()['labels'])>0:
                self.labels = metadata.class_vocab()['labels'][self.idxs]
            else:
                self.labels = []
        
        # FETCH THE KWARGS FROM THE YAML
        self.fetch_kwargs(kwargs)

        # LVD: A model, with a learnable context, and a query method for novel PCs
        if self.paradigm == "LoVD":  # LoVD: A model, with a learnable context, and a query method for novel PCs
            if self.powerup==1:
                self.model = Network_LoVD(self.size_layers, self.gt_points*3, res=self.occ_res, input_dim=self.input_dim , b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]), segm=self.segm, labels=self.labels)            
            else:
                self.model = Network_LoVD(self.size_layers, self.gt_points*3, res=self.occ_res, input_dim=self.input_dim , b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]), selfsup=self.selfsup, segm=self.segm)
                
        # Universal embedding Baseline
        if self.paradigm == "Uni":  # Universal embedding: A network for incoming PC and one network for the Template
            self.model = Network_LoVD(self.size_layers, self.gt_points*3, res=self.occ_res, input_dim=self.input_dim , b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]), selfsup=self.selfsup, segm=self.segm, labels=self.labels, paradigm = 'Uni')

        # LIE Baseline
        if self.paradigm == "LIE":  # LIE: We learn basis and Descriptors
            self.models = {}
            self.models['basis'] = Network_LoVD(self.size_layers, self.n_basis, res=self.occ_res, input_dim=self.input_dim , b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]), selfsup=self.selfsup, segm=self.segm, labels=self.labels).cuda()
            self.models['desc']  = Network_LoVD(self.size_layers, self.n_desc, res=self.occ_res, input_dim=self.input_dim , b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]), selfsup=self.selfsup, segm=self.segm, labels=self.labels).cuda()

            self.model = self.models['basis'] 
            self.flag_mode = 'basis'
        
   
   # USEFUL FUNCTION FOR LIE BASELINE
    def change_mode(self,mode):
        if mode not in ['basis','desc']:
            raise RuntimeError('wrong name')
            
        self.flag_mode = mode
        self.model = self.models[mode]     
            
    def forward(self, x) -> torch.Tensor:
        if self.paradigm == "Uni":
            return self.model(x)
        if self.paradigm == "LVD":
            return self.model(x)
        if self.paradigm == "LIE":
            return self.model(x)
        
    def step(self, batch, test=False) -> Mapping[str, Any]:
        self.count_steps += 1
        
        # FETCH VOXELS, SAMPLED POINTS, TARGET GT
        self._input_voxels = batch['input_voxels'].float().to(self.dev)
        self._input_points = batch['input_points'].float().to(self.dev)
        self._target_smpl  = batch['smpl_vertices'].float().to(self.dev)
        _B = self._input_voxels.shape[0]
        
        self._input_voxels = torch.cat((torch.clamp(self._input_voxels, 0, 0.01)*100,
                                        torch.clamp(self._input_voxels, 0, 0.02)*50,
                                        torch.clamp(self._input_voxels, 0, 0.05)*20,
                                        torch.clamp(self._input_voxels, 0, 0.10)*20,
                                        torch.clamp(self._input_voxels, 0, 0.15)*15,
                                        torch.clamp(self._input_voxels, 0, 0.20)*10,
                                        
                                        torch.gradient(self._input_voxels,axis=2)[0]*self.grad,
                                        torch.gradient(self._input_voxels,axis=3)[0]*self.grad,
                                        torch.gradient(self._input_voxels,axis=4)[0]*self.grad,
                                        
                                        torch.clamp((torch.abs(torch.gradient(self._input_voxels,axis=2)[0]) + torch.abs(torch.gradient(self._input_voxels,axis=3)[0]) + torch.abs(torch.gradient(self._input_voxels,axis=4)[0])),-0.1,0.1)*self.grad,
                                        torch.clamp((torch.abs(torch.gradient(self._input_voxels,axis=2)[0]) + torch.abs(torch.gradient(self._input_voxels,axis=3)[0]) + torch.abs(torch.gradient(self._input_voxels,axis=4)[0])),-0.1,0.1)*self.grad,
                                        torch.clamp((torch.abs(torch.gradient(self._input_voxels,axis=2)[0]) + torch.abs(torch.gradient(self._input_voxels,axis=3)[0]) + torch.abs(torch.gradient(self._input_voxels,axis=4)[0])),-0.1,0.1)*self.grad,
                                        
                                        self._input_voxels
                                        ), 1)
        
         # Not needed, but if needed, you can reshape the input_voxels
        if  self._input_voxels.shape[2]==64 or self._input_voxels.shape[2]==128:
            self._input_voxels2 = self._input_voxels 
        elif 64**3 == self._input_voxels.shape[2]:
           self._input_voxels2 = torch.reshape(self._input_voxels,(_B,self._input_voxels.shape[1],64,64,64)) 
        else:
           self._input_voxels2 = torch.reshape(self._input_voxels,(_B,self._input_voxels.shape[1],128,128,128))

        #### HOW TRAINING AND TEST SHOULD BE PERFORMED  ####
        
        #### START LoVD ####
        if self.paradigm == "LoVD":
            log = {}
            
            # FIRST, WE COMPUTE THE GT VERTEX DESCENT, NO GRAD NEEDED
            with torch.no_grad():
                
                # COMPUTE THE DISTANCE OF EVERY SAMPLED POINT TO EVERY TARGET POINT
                dist = torch.unsqueeze(self._input_points ,1) - torch.unsqueeze(self._target_smpl,2)
                clamp = self.clamp
                
                # WE CLAMP THE GT OFFSET WITHIN THE DECIDED VALUE
                if self.clamp_style == 0:
                    dist = torch.clip(dist, -1*clamp, clamp)
                if self.clamp_style == 1:
                    norms = torch.linalg.norm(dist,2,axis=-1,keepdim=True)
                    factors = torch.clip(norms, 0, self.max_norm) / norms
                    dist = dist * factors
            
            # EXTRACT GLOBAL FEATURES        
            self.model(self._input_voxels2)
            _numpoints = self._input_points.shape[1]

            # Predict the LVD for the sampled points
            pred = self.model.query(self._input_points)
            pred = pred.reshape(_B, self.gt_points, 3, _numpoints).permute(0, 1, 3, 2)
            
            # L1 error
            self._loss_L2 = torch.abs(dist - pred)
            self._loss_L2 = self._loss_L2.mean()
                        
                    
            log["loss_l2"] = self._loss_L2
            self._loss_tot = self._loss_L2

            # COMPLETE LOSS TO LOG    
            log["loss"] = self._loss_tot
            
            if test:
                self._target_smpl_full = batch['smpl_vertices_full'].float().to(self.dev)
                
                rec_list = []
                with torch.no_grad():
                    input_points = torch.zeros(1, self.gt_points, 3).to(self.dev)
                    _B = 1
                    iters = 30
                    inds = np.arange(self.gt_points)
                    self.model(self._input_voxels2[0:1])
                    for it in range(iters):
                        if self.unsup:  
                            pred_dist,dist_func = self.model.query(input_points)
                        else:
                            pred_dist = self.model.query(input_points)
                        pred_dist = pred_dist.reshape(_B, self.gt_points, 3, -1).permute(0, 1, 3, 2)
                        input_points = - pred_dist[:, inds, inds] + input_points
                        rec_error = torch.sum(torch.abs(input_points[0] - self._target_smpl[0]))/self._target_smpl[0].shape[0]
                    rec_list.append(rec_error)
                    pred_mesh = input_points[0].cpu().data.numpy()
                    m = self.match(self._target_smpl[0], input_points[0])
                    src, tar, both = pcs_renderings(self._target_smpl[0:1], input_points[0:1], np.expand_dims(m,0))
                    log["loss_rec"] = rec_list
                    log["both"] = both
            return log

        #### END LVD   ####

        ## START UNIVERSAL  ##
        if self.paradigm == "Uni":
            log = {}

            # Predict the universal point-wise embedding
            self.model(self._input_voxels2)
            pred = self.model.query(self._target_smpl)

            pred_src = torch.transpose(pred[:-1, :, :],1,2)
            pred_tar = torch.transpose(pred[1:, :, :],1,2)
            tar =  self._target_smpl[1:]
            src = self._target_smpl[:-1]
            
            # Compute Loss
            loss = loss_uni(pred_src, pred_tar, tar)
            log["loss_l2"] = loss
            log["loss"] = loss
            
            if test:
                with torch.no_grad():
                    m = self.match(pred_src, pred_tar)
                    ct_np = tar.detach().cpu().numpy()
                    rec_errs = [np.sqrt(np.sum((ct_np[i, m[1][i]] - ct_np[i]) ** 2)) for i in range(ct_np.shape[0])]
                    m_error = np.mean(rec_errs)
                    src, tar, both = pcs_renderings(src, tar, m)
                    log["loss_rec"] = rec_errs
                    log["both"] = both                    
            return log

        ## END UNIVERSAL  ##
        
    def set_input(self, input):
        self._input_voxels = input['input_voxels'].float().to(self.dev)
        self._input_points = input['input_points'].float().to(self.dev)
        self._target_smpl  = input['smpl_vertices'].float().to(self.dev)

        self._input_voxels = torch.cat((torch.clamp(self._input_voxels, 0, 0.01)*100,
                                        torch.clamp(self._input_voxels, 0, 0.02)*50,
                                        torch.clamp(self._input_voxels, 0, 0.05)*20,
                                        torch.clamp(self._input_voxels, 0, 0.1)*20,
                                        torch.clamp(self._input_voxels, 0, 0.15)*15,
                                        torch.clamp(self._input_voxels, 0, 0.2)*10,
                                        self._input_voxels
                                        ), 1)
        return 

    # KNN MATCH -> USED ONLY IN BASELINES
    def match(self, src, tar):
        # For each point of tar, find a point on src
        c = torch.cdist(src, tar)
        match = torch.min(c, 1)
        m = match[1].detach().cpu().numpy()
        return m
        
    # THIS IS WHAT TO DO IF YOU ARE IN TRAINING    
    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:

        # Step
        step_out = self.step(batch)
        if self.clamp_style==2:
            self.log_dict(
                {"loss/train_lclamp": step_out["loss_clamp"].cpu().detach(),},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )             
        # Log
        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach(),
             "loss/train_l2": step_out["loss_l2"].cpu().detach(),},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    # THIS IS WHAT TO DO IF YOU ARE IN VALIDATION
    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # Parse the data

        # Step
        with torch.no_grad():
            step_out = self.step(batch)

        # Log
        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return step_out

    # THIS IS WHAT TO DO IF YOU ARE IN TEST
    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # Parse the data

        # Step
        step_out = self.step(batch, test=True)


        # Log
        if self.paradigm == "LVD":
            # Error for the 3 steps
            wandb.log({"test/rec_fin": step_out["loss_rec"][0]})
        # Correspondence
        wandb.log({"loss/test": step_out["loss"].cpu().detach()})

        fig = render_result(step_out["both"][0])
        wandb.log({"test/test": fig})

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

