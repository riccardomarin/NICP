# Automatic building of cython extensions
import numpy as np
import pyximport
pyximport.install(
    setup_args={
        "include_dirs": [np.get_include(), "./utils/libvoxelize"], 
        "script_args": ["--cython-cplus"]
    }, reload_support=True, language_level=3
)

import argparse
import os
import traceback
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool

import torch
import trimesh
from tqdm import tqdm

from utils.voxels import VoxelGrid
from utils.parallel_map import parallel_map

## SET RESOLUTION OF THE VOXELIZATION
res = 64 

faces = np.load("./assets/faces.npy")


### VOXELIZATION UTILS  ####

def voxelize(path, res):
    output_file = os.path.dirname(path) + '/vox_{}.npy'.format(res)
    try:
        if os.path.exists(output_file):
            return

        mesh = trimesh.load(path , process=False)
        occupancies = VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(output_file, occupancies)

    except Exception as err:
        path = os.path.normpath(path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    print('finished {}'.format(path))


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

def voxelize_distance(v, res, mesh_faces,OUT_PATH_OCC,OUT_PATH_SCAL ):      
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
    
    ## Save voxels and aligned vertices    
    torch.save(voxels,OUT_PATH_OCC / str(f'{idx:09}.pt') )
    torch.save(vertices,OUT_PATH_SCAL/ str(f'{idx:09}.pt')  )
    
##############

def main(cfg):

    # needed to support cuda in parallel_map
    torch.multiprocessing.set_start_method('spawn', force=True)

    tqdm_bar = tqdm(cfg.datasets, total=len(cfg.datasets), ncols=80)
    for DATASET in tqdm_bar:
        tqdm_bar.set_description(f"Processing {DATASET}")
        INPUT_PATH = args.input_path / cfg.exp / 'stage_III' / DATASET / 'data_v.pt'
        OUT_PATH_OCC = args.input_path / cfg.exp / 'stage_III' / DATASET / 'ifnet_indi' / "occ_dist"
        OUT_PATH_SCAL = args.input_path / cfg.exp /'stage_III' / DATASET / 'ifnet_indi' / "verts_occ_dist"

        OUT_PATH_OCC.mkdir(parents=True, exist_ok=True)
        OUT_PATH_SCAL.mkdir(parents=True, exist_ok=True)

        k = torch.load(INPUT_PATH)
        k = k.reshape((k.shape[0],k.shape[1]//3,-1),1)
        id = np.expand_dims(np.tile(np.arange(k.shape[0]),(k.shape[1],1)).T, axis=-1)
        k = np.dstack((k,id))
        # k = [(i, k[i]) for i in range(k.shape[0])]
        
        # use single queue to collect results on the fly
        manager = mp.Manager()
        # collector = mp.Process(target=write_output, args=(len(k), cfg.res, OUT_PATH_OCC, OUT_PATH_SCAL))
        # collector.start()
        parallel_map(k, voxelize_distance, n_jobs=args.jobs, const_args={
        "mesh_faces": faces,
        "res": cfg.res,
        "OUT_PATH_OCC" :OUT_PATH_OCC,
        "OUT_PATH_SCAL" : OUT_PATH_SCAL
    }, tqdm_kwargs={"leave": False}
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data voxelization")

    parser.add_argument("--exp", "-e", type=str, default='V1_SV1_T5', help="Experiment name")
    parser.add_argument("--datasets", "-d", type=str, default='vald', nargs="+", choices=["train", "vald", "test"], help="Dataset name")
    parser.add_argument("--input_path", "-i", type=Path, default=Path('/mnt/sda/data_p/'), help="Path to input folder")
    parser.add_argument("--jobs", "-j", type=int, default=4, help="Number of parallel jobs")

    args = parser.parse_args()
    args.datasets = list(set(args.datasets))

    main(args)

