import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import hydra
import numpy as np
import pytorch_lightning as pl
from hydra import compose, initialize
import open3d  as o3d
import time   
import tqdm
import torch
import trimesh
import sys 

from utils_cop.prior import SMPLifyAnglePrior, MaxMixturePrior
from utils_cop.SMPL import SMPL
from lvd_templ.paths import chk_pts, home_dir, output_dir as out_folder, path_demo

os.chdir(home_dir)

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO

from lvd_templ.data.datamodule_AMASS import MetaData
from lvd_templ.evaluation.utils import vox_scan, fit_LVD, selfsup_ref, SMPL_fitting, fit_plus_D, fit_cham

from omegaconf import DictConfig
import gc

from io import StringIO
import streamlit.components.v1 as components

import open3d as o3d
import plotly.graph_objects as go  
import calendar
import time


sys.path.append('.')
sys.path.append('./src')
sys.path.append('./preprocess_voxels')


device = torch.device('cuda')
import gc 

import warnings
warnings.filterwarnings("ignore")

def export_mesh(T, r,t,s, path):
        T.apply_scale(s)
        T.apply_translation(t)
        T.apply_transform(r)
        
        k = T.export(path)
        return


def get_dataset(name):
    if name=='demo':
        return path_demo
                        
    raise ValueError('this challenge does not exists')


def get_model(chk):
    chk_zip = glob.glob(chk + 'checkpoints/*.zip')[0]
    ### Restore the network
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../../../" + str(chk))
    cfg_model = compose(config_name="config")
    # print(OmegaConf.to_yaml(cfg_model))
    train_data = hydra.utils.instantiate(cfg_model.nn.data.datasets.train, mode="test")
    
    MD = MetaData(class_vocab=train_data.class_vocab)
    model: pl.LightningModule = hydra.utils.instantiate(cfg_model.nn.module, _recursive_=False, metadata=MD)
    old_checkpoint = NNCheckpointIO.load(path=chk_zip)
    
    module = model._load_model_state(checkpoint=old_checkpoint, metadata=MD).to(device)
    module.model.eval()
    
    return module, MD, train_data, cfg_model

def plot_mesh(m: trimesh, showscale, **kwargs):
    """
    Plot the mesh in a plotly graph object
    :param m: the mesh to plot
    :param kwargs: possibly additional parameters for the go.Mesh3D class
    :return: the plotted mesh
    """
    vertices = m.vertices.astype(np.float64)
    if hasattr(m, 'faces'):

        faces = m.faces.astype(np.uint32)
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            colorscale="Viridis",
            opacity=1,
            showscale=showscale,
            **kwargs,
        )
    else:
        return go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode = 'markers',
            marker_size = 1
        )


os.chdir(home_dir)
model_name = '1ljjfnbx'
chk = chk_pts + model_name + '/'
model_name = model_name

alpha = np.pi/2 
origin, xaxis = [0, 0, 0], [1, 0, 0]

SMPL_model = SMPL('./neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).cuda()
prior = MaxMixturePrior(prior_folder='utils_cop/prior/', num_gaussians=8) 
prior.to(device)

module, MD, train_data, cfg_model = get_model(chk)
module.cuda()

Rx = trimesh.transformations.rotation_matrix(alpha, xaxis)

st.write('# Welcome to NSR Demo!')
st.write('Use the box below to upload a 3D mesh in .ply.')
uploaded_file = st.file_uploader("Choose a file")


current_GMT = time.gmtime()

if uploaded_file is not None:
    # To read file as bytes:
    time_stamp = calendar.timegm(current_GMT)
    os.mkdir(os.path.join("streamlitTempDir", str(time_stamp)))
    print(os.path.join("streamlitTempDir", str(time_stamp)))
    bytes_data = uploaded_file.getvalue()

    with open(os.path.join("streamlitTempDir", str(time_stamp),uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
        
    T = o3d.io.read_triangle_mesh(os.path.join("streamlitTempDir",uploaded_file.name))    
    o3d.io.write_triangle_mesh(os.path.join("streamlitTempDir", str(time_stamp),"input.obj"),T)
    
    scan_src = trimesh.load(os.path.join("streamlitTempDir", str(time_stamp),uploaded_file.name), process=False, maintain_order=True)

    a = plot_mesh(scan_src,True)

    fig = go.Figure(data=[a])
    fig['layout']['scene']['aspectmode'] = "data"
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    res = MD.class_vocab()['occ_res']
    gt_points = MD.class_vocab()['gt_points']
    gt_idxs = train_data.idxs
    type = cfg_model['nn']['data']['datasets']['type']
    grad = cfg_model['nn']['module']['grad']

    start = time.time()

    st.write("### Processing - Voxelizing your input")
    scan_src.apply_transform(Rx)
    voxel_src, mesh_src, scale, trasl = vox_scan(scan_src, res, style=type, grad=grad)
    st.write("Done! It took " + "{:.2f}".format((time.time() - start)) + " secs")
    mesh_src.export(os.path.join("streamlitTempDir", str(time_stamp), 'aligned.ply'))
    
    st.write("### Running N-ICP to refine it.")
    start = time.time()
    module.train()
    selfsup_ref(module, torch.tensor(np.asarray(scan_src.vertices)), voxel_src, gt_points,steps=20, lr_opt=0.00001)
    module.eval()
    st.write("Done! It took " + "{:.2f}".format((time.time() - start)) + " secs" )
    
    st.write("### NF Convergence + SMPL fitting ... (~30 seconds)")
    start = time.time()
    reg_src =  fit_LVD(module, gt_points, voxel_src, iters=50)
    out_s, params = SMPL_fitting(SMPL_model, reg_src, gt_idxs, prior, iterations=2000)

    SMPL_backbone_OUR = trimesh.Trimesh(out_s,SMPL_model.faces)
    st.write("Done! It took " + "{:.2f}".format((time.time() - start)) + " secs" )
    
    st.write("Here is our output")
    c = plot_mesh(SMPL_backbone_OUR,True)
    fig3 = go.Figure(data=[c])
    fig3['layout']['scene']['aspectmode'] = "data"
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
    
    SMPL_backbone_OUR.export(os.path.join("streamlitTempDir", str(time_stamp), 'our.ply'))
    
    st.write("### Running Chamfer refinement ... (~10 seconds)")
    start = time.time()
    out_cham_s, params = fit_cham(SMPL_model, out_s, mesh_src.vertices, prior,params)
    st.write("Done! It took " + "{:.2f}".format((time.time() - start)) + " secs" )
    
    SMPL_backbone_OUR = trimesh.Trimesh(out_cham_s,SMPL_model.faces)
    SMPL_backbone_OUR.export(os.path.join("streamlitTempDir", str(time_stamp), 'our_ch.ply'))
    c = plot_mesh(SMPL_backbone_OUR,True)
    fig3 = go.Figure(data=[c])
    fig3['layout']['scene']['aspectmode'] = "data"
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)    
    
    for p in params.keys():
        params[p] = params[p].detach().cpu().numpy()            
    np.save(os.path.join("streamlitTempDir", str(time_stamp), 'params.npy'),params)
    
    
