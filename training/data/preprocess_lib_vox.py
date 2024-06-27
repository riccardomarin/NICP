import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import numpy as np
from functools import partial
import traceback
from preprocess_voxels import voxels
import sys 

from scipy.spatial import cKDTree as KDTree

import random

sys.path.append("/home/ubutnu/Documents/Projects/LVD_templ/lvd_templ/data")
sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "./preprocess_voxels/")) 
import preprocess_voxels.implicit_waterproofing as iw

def to_off(path):
    mesh_name = os.path.basename(path)
    output_file = OUT_PATH + '/' + mesh_name[:-4]  + '/off_mesh.off'

    if not(os.path.exists(OUT_PATH + '/' + mesh_name[:-4]  + '/')):
        os.mkdir(OUT_PATH + '/' + mesh_name[:-4]  + '/')

    if os.path.exists(output_file):
        return

    input_file  = INPUT_PATH + '/' + mesh_name

    cmd = 'meshlabserver -i {} -o {}'.format(input_file,output_file)
    os.system(cmd)

def scale(path):
    output_file = os.path.dirname(path)  + '/scaled_off_mesh.off'

    try:
        mesh = trimesh.load(os.path.dirname(path) + '/off_mesh.off', process=False)
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

def voxelized_pointcloud_sampling(path, res, num_points, grid_points, kdtree):
    try:
        out_file =  os.path.dirname(path) + '/vox_pc_{}res_{}points.npz'.format(res, num_points)

        if os.path.exists(out_file):
            print('File exists. Done.')
            return

        mesh = trimesh.load(path)
        point_cloud = mesh.sample(num_points)


        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)


        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = bb_min, bb_max = bb_max, res = res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def boundary_sampling(path, sigma, sample_num):
    try:
        filename = os.path.dirname(path)+ '/boundary_{}_samples.npz'.format(sigma)

        if os.path.exists(filename):
            return

        off_path = path
        out_file = filename

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

DATASET = './data/faust_new'
INPUT_PATH = './' + DATASET + '/raw'
OUT_PATH = './' + DATASET + '/processed'

bb_min = -0.5
bb_max = 0.5

# # Converting to .off
p = Pool(mp.cpu_count())
p.map(to_off, glob.glob(INPUT_PATH + '/*.ply'))

# # Scaling
p.map(scale, glob.glob( OUT_PATH + '/*/off_mesh.off'))

# # Voxelizing
p.map(partial(voxelize, res=32), glob.glob(OUT_PATH + '/*/scaled_off_mesh.off'))
p.map(partial(voxelize, res=128), glob.glob(OUT_PATH + '/*/scaled_off_mesh.off'))

# # PC Voxelizing
# res = 128
# num_points = 3000

# grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, res)
# kdtree = KDTree(grid_points)

# paths = glob.glob( OUT_PATH + '/*/scaled_off_mesh.off')

# p.map(partial(voxelized_pointcloud_sampling, res=res, num_points=num_points, grid_points=grid_points, kdtree=kdtree), paths)

# sample_num = 100000

# p.map(partial(boundary_sampling, sigma=0.1, sample_num=sample_num),  glob.glob(OUT_PATH +'/*/scaled_off_mesh.off'))
# p.map(partial(boundary_sampling, sigma=0.01, sample_num=sample_num),  glob.glob(OUT_PATH +'/*/scaled_off_mesh.off'))



# def create_voxel_off(path):


#     voxel_path = path + '/voxelization_{}.npy'.format( res)
#     off_path = path + '/voxelization_{}.off'.format( res)


#     if unpackbits:
#         occ = np.unpackbits(np.load(voxel_path))
#         voxels = np.reshape(occ, (res,)*3)
#     else:
#         voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

#     loc = ((min+max)/2, )*3
#     scale = max - min
    

# sys.path.append("/home/ubutnu/Documents/Projects/LVD_templ/lvd_templ/data/preprocess_voxels") 
# sys.path.append("/home/ubutnu/Documents/Projects/LVD_templ/lvd_templ/data/") 
# sys.path.append("/home/ubutnu/Documents/Projects/LVD_templ/lvd_templ/data/preprocess_voxels/libvoxelize") 
# from voxels import VoxelGrid
# VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
# print('Finished: {}'.format(path))

# unpackbits = True
# res = args.res
# min = -0.5
# max = 0.5

# p = Pool(mp.cpu_count())
# p.map(create_voxel_off, glob.glob( ROOT + '/*/*/'))