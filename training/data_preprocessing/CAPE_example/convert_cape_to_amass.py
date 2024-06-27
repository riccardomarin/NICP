import numpy as np 

cape = np.load('/home/ubutnu/Documents/Projects/PTF/data/CAPE_sampling-rate-5/00032/longshort_ATUsquat/longshort_ATUsquat.000146.npz')
amass =  np.load('/mnt/sda/Projects/AMASS/support_data/amass_npz/ACCAD/ACCAD/Female1General_c3d/A1 - Stand_poses.npz')


for f in amass:
    print(f)
    
for f in cape:
    print(f)  
    

amass['trans'].shape
cape['trans'].shape

amass['gender']
cape['gender']

amass['betas'].shape
cape['pose_body'].shape

amass['poses'].shape
cape['pose_body'].shape
cape['pose_hand'].shape

import open3d as o3d

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cape['vertices_scaled']))

o3d.visualization.draw([pc])

cape['a_pose_mesh_points']

#####

import torch 

res = 64
type = 'occ_dist'

EXP = 'V1_SV1_T5'
DATASET = 'train'
INPUT_PATH = '/mnt/sda/data_p/' + EXP +'/stage_III/' + DATASET + '/data_v.pt'

k = torch.load(INPUT_PATH)


data_v.extend(np.reshape(body_v.detach().cpu().numpy().astype(np.float32),(-1,body_v.shape[1]*body_v.shape[2])))

#####
import glob 
import tqdm 

lista = ['00032', '00096', '00127', '00134', '00145', '02474', '03223', '03284', '03331', '03375', '03383', '03394']
data_v = []

for l in tqdm.tqdm(lista):
    frames = glob.glob('/home/ubutnu/Documents/Projects/PTF/data/CAPE_sampling-rate-5/' + l + '/*/*.npz')
    print('l')
    for f in frames:
        dd = np.load(f)
        points = dd['vertices_scaled']
        data_v.extend(np.reshape(points.astype(np.float32),(1, points.shape[0]*points.shape[1])))


torch.save(torch.tensor(np.asarray(data_v, np.float32)),'data_v.pt')
