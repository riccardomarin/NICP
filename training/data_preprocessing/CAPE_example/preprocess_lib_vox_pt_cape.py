
from __future__ import division

import sys 
import os

import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
from functools import partial
import traceback
import trimesh
#import preprocess_voxels.implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
from os import path as osp
import random

import torch 
from data_processing import voxels

support_dir = './amass/support_data/'
bm_fname = osp.join(support_dir, 'body_models/neutral/model.npz')
dict = np.load(bm_fname)

faces = np.asarray(dict['f'])

def scale(path):
    output_file = os.path.dirname(path)  + '/scaled_off_mesh.off'

    try:
        mesh = trimesh.load(os.path.dirname(path) + '/off_mesh.off', process=False)
        mesh.faces = M.faces
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) /2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1/total_size)
        mesh.export(output_file)
    except:
        print('Error with {}'.format(output_file))
    print('Finished {}'.format(output_file))

def voxelize(path, res):
    output_file = os.path.dirname(path) + '/vox_{}.npy'.format(res)
    try:
        if os.path.exists(output_file):
            return

        mesh = trimesh.load(path , process=False)
        occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(output_file, occupancies)

    except Exception as err:
        path = os.path.normpath(path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    print('finished {}'.format(path))

occ = {}

def write_ply(v):
    coord = v[:, 0:3]
    idx = int(v[:, 3][0])
    
    mesh = trimesh.Trimesh(vertices = coord,faces = faces)
    
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) /2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1/total_size)
       
    occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
    occupancies = np.reshape(occupancies, -1)

    if not occupancies.any():
        raise ValueError('No empty voxel grids allowed.')

    occupancies = np.packbits(occupancies)


    torch.save(occupancies,OUT_PATH_OCC + '/' + str(f'{idx:09}') + '.pt' )
    torch.save(mesh.vertices,OUT_PATH_SCAL+ '/' + str(f'{idx:09}') + '.pt' )
    
#####
def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix

def voxelize_distance(v):
    vertices = v[:, 0:3]
    idx = int(v[:, 3][0])
    
    mesh = trimesh.Trimesh(vertices = vertices,faces = faces)
    
    resolution = res # Voxel resolution
    b_min = np.array([-0.8, -0.8, -0.8]) 
    b_max = np.array([0.8, 0.8, 0.8])
    step = 5000
    
    
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) /2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1/total_size)
    
    vertices = mesh.vertices
    factor = max(1, int(len(vertices) / 20000)) # We will subsample vertices when there's too many in a scan !

    with torch.no_grad():
        v = torch.FloatTensor(vertices).cuda()
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        points_npy = coords.reshape(3, -1).T
        iters = len(points)//step + 1

        all_distances = []
        for it in range(iters):
            it_v = points[it*step:(it+1)*step]
            distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1))**2).sum(-1)
            distance = distance.min(0)[0].cpu().data.numpy()
            all_distances.append(distance)
        signed_distance = np.concatenate(all_distances)
    del v 
    del coords 
    
    voxels = signed_distance.reshape(resolution, resolution, resolution)    
    torch.save(voxels,OUT_PATH_OCC + '/' + str(f'{idx:09}') + '.pt' )
    torch.save(vertices,OUT_PATH_SCAL+ '/' + str(f'{idx:09}') + '.pt' )
    
    
#######

res = 64
type = 'occ_dist'

EXP = 'V1_SV1_T8'
DATASET = 'train'

INPUT_PATH = '/home/ubutnu/Documents/Projects/NFICP_preprocess/data_v.pt'
print(INPUT_PATH)
OUT_PATH_OCC = '/mnt/sda/cape/' + EXP +'/stage_III/' + DATASET + '/ifnet_indi/' + type + '_'+ str(res)+ '/'
OUT_PATH_SCAL = '/mnt/sda/cape/' + EXP +'/stage_III/' + DATASET + '/ifnet_indi/verts_' + type + '/'

if not(os.path.exists('/mnt/sda/cape/' + EXP +'/stage_III/' + DATASET + '/ifnet_indi/')):
    os.mkdir('/mnt/sda/cape/' + EXP +'/stage_III/' + DATASET + '/ifnet_indi/')

if not(os.path.exists(OUT_PATH_OCC)):
    os.mkdir(OUT_PATH_OCC)
if not(os.path.exists(OUT_PATH_SCAL)):
    os.mkdir(OUT_PATH_SCAL)



k = torch.load(INPUT_PATH)

k = k.reshape((k.shape[0],k.shape[1]//3,-1),1)
id = np.expand_dims(np.tile(np.arange(k.shape[0]),(k.shape[1],1)).T, axis=-1)
k = np.dstack((k,id))

from multiprocessing import Manager    
from torch.multiprocessing import Pool, Process, set_start_method
  
manager = Manager()
shared_list = manager.dict()
data_v_scal = manager.dict()
processes = []

# voxelize_distance(k[0])
# voxelize_distance(k[1])

with Pool(4) as p: 
    p.map(voxelize_distance, k)
    
    

beta = torch.tensor(k[:,0:1,0])
torch.save(torch.tensor(np.asarray(beta, np.float32)), "/mnt/sda/cape/V1_SV1_T8/stage_III/train/beta.pt")


