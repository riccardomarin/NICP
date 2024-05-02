
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import sys
import tqdm 
from typing import Union
from typing import Optional

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./preprocess_voxels')
sys.path.append("./data/preprocess_voxels/libvoxelize")
sys.path.append("./data/preprocess_voxels/")
sys.path.append("./data/")

from preprocess_voxels import voxels
#from chamferdist import ChamferDistance

class OptimizationSMPL(torch.nn.Module):
    def __init__(self):
        super(OptimizationSMPL, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 72).cuda())
        self.beta = torch.nn.Parameter((torch.zeros(1, 300).cuda()))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())
        self.scale = torch.nn.Parameter(torch.ones(1).cuda()*1)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale

from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input


## Utility Functions for Chamfer Distance
def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights,
    batch_reduction: Union[str, None],
    point_reduction: Union[str, None],
    norm: int,
    abs_cosine: bool,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if batch_reduction == "mean":
                div = weights.sum() if weights is not None else max(N, 1)
                cham_x /= div
                if return_normals:
                    cham_norm_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_normals is from one minus the cosine similarity.
            If True (default), loss_normals is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite normals are considered
            equivalent to exactly matching normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        return cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_normals,
            x_normals,
            weights,
            batch_reduction,
            point_reduction,
            norm,
            abs_cosine,
        )
        if point_reduction is not None:
            return (
                cham_x + cham_y,
                (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
            )
        return (
            (cham_x, cham_y),
            (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None,
        )



class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        bidirectional: Optional[bool] = False,
        reverse: Optional[bool] = False,
        batch_reduction: Optional[str] = "mean",
        point_reduction: Optional[str] = "sum",
    ):
        if reverse:
            _source = target_cloud
            _target = source_cloud
        else:
            _source = source_cloud
            _target = target_cloud
        return chamfer_distance(
            _source, _target,
            single_directional= not bidirectional,
            batch_reduction=batch_reduction, 
            point_reduction=point_reduction)[0]

################# Voxalize Function
def voxelize(mesh, res):
    # Transform into a grid
    occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
    occupancies = np.reshape(occupancies, -1)

    if not occupancies.any():
        raise ValueError('No empty voxel grids allowed.')

    return occupancies, mesh




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

def voxelize_distance(scan, res):
    resolution = res # Voxel resolution
    b_min = np.array([-0.8, -0.8, -0.8]) 
    b_max = np.array([0.8, 0.8, 0.8])
    step = 5000

    vertices = scan.vertices
    factor = max(1, int(len(vertices) / 20000)) # We will subsample vertices when there's too many in a scan !

    # NOTE: It was easier and faster to just get distance to vertices, instead of voxels carrying inside/outside information,
    # which will only be possible for closed watertight meshes.
    with torch.no_grad():
        v = torch.FloatTensor(vertices).cuda()
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        points_npy = coords.reshape(3, -1).T
        iters = len(points)//step + 1

        all_distances = []
        for it in range(iters):
            it_v = points[it*step:(it+1)*step]
            it_v_npy = points_npy[it*step:(it+1)*step]
            distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1))**2).sum(-1)
            #contain = scan.contains(it_v_npy)
            distance = distance.min(0)[0].cpu().data.numpy()
            all_distances.append(distance)
        #contains = scan.contains(points_npy)
        signed_distance = np.concatenate(all_distances)

    voxels = signed_distance.reshape(resolution, resolution, resolution)
    return voxels, scan

#######

# Fit SMPL to the LVD prediction
def SMPL_fitting(SMPL_model, in_points, gt_idxs, prior, iterations = 1000):
    # Hyperparameters
    factor_beta_reg = 0.01
    factor_pose_reg = 0.00000001
    lr = 1e-1
    lr_eps = 1e-5
    
    # Setup the optimization
    parameters_smpl = OptimizationSMPL().cuda()
    optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters())
    pred_mesh_torch = torch.FloatTensor(in_points).cuda()
    
    # SMPL FITTING
    for i in tqdm.tqdm(range(iterations), desc="FIT SMPL TO NF PREDICTION"):
        # Forward pass
        pose, beta, trans, scale = parameters_smpl.forward()
        vertices_smpl = (SMPL_model.forward(theta=pose, beta=beta, get_skin=True)[0][0] + trans)*scale
        distances = torch.abs(pred_mesh_torch - vertices_smpl[gt_idxs])
        
        # Get Losses
        loss = distances.mean()
        prior_loss = prior.forward(pose[:, 3:], None)
        beta_loss = (beta**2).mean()
        loss = loss + prior_loss*factor_pose_reg + beta_loss*factor_beta_reg
        
        # Optimization
        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()
        for param_group in optimizer_smpl.param_groups:
            param_group['lr'] = lr*(iterations-i)/iterations + lr_eps

    # Obtain model and parameter
    with torch.no_grad():
        pose, beta, trans, scale = parameters_smpl.forward()
        vertices_smpl = (SMPL_model.forward(theta=pose, beta=beta, get_skin=True)[0][0] + trans)*scale
        fit_mesh = vertices_smpl.cpu().data.numpy()
        params = {}
        params['loss'] = loss.detach()
        params['beta'] = beta
        params['pose'] = pose
        params['trans'] = trans
        params['scale'] = scale
        
    return fit_mesh, params


# def get_chamfer_dist(ref_vertices, points):
#     step = 100
#     iters = len(points)//step
#     if len(points)%step != 0:
#         iters += 1
#     if not torch.is_tensor(points):
#         points = torch.FloatTensor(points)
#     if not torch.is_tensor(ref_vertices):
#         ref_vertices = torch.FloatTensor(ref_vertices)
#     points = points.cuda()
#     ref_vertices = ref_vertices.cuda()
#     with torch.no_grad():
#         all_dists = []
#         for i in range(iters):
#             dist = ((ref_vertices.unsqueeze(0) - points[i*step:(i+1)*step].unsqueeze(1))**2).sum(-1)
#             all_dists.append(torch.sqrt(dist.min(1)[0]))
#         all_dists = torch.cat(all_dists)
#     return all_dists


#### SELF SUPERVISED NF ICP
def selfsup_ref(module, input_points, voxel_src, gt_points,steps=10, lr_opt=0.00001):
    optimizer = torch.optim.Adam(module.parameters(), lr=lr_opt)

    for i in tqdm.tqdm(np.arange(0,steps),desc="NF-ICP"):
        # Sample points on the target surface
        with torch.no_grad():
            factor = max(1, int(input_points.shape[0] / 20000))
            input_points = input_points[torch.randperm(input_points.size()[0])]
            input_points_res = input_points[1:input_points.shape[0]:factor,:].type(torch.float32).unsqueeze(0).cuda()

        optimizer.zero_grad()
        
        # Extract Features
        module.model(voxel_src)
        
        # Query points on the target surface
        pred_dist = module.model.query(input_points_res)
        res = type(pred_dist) is tuple
        if res:
            pred_dist = pred_dist[0]
        
        pred_dist = pred_dist.reshape(1, gt_points, 3, -1).permute(0, 1, 3, 2)
        
        # Collect the offset with the minimum norm for each target vertex
        v, _ = torch.min(torch.sum(pred_dist**2,axis=3),axis=1)
        
        # Global loss
        loss = torch.sum(v)
        
        # Optimize
        loss.backward()
        optimizer.step()

def fit_cham(SMPL_model, pred_mesh, vertices_scan, prior,init, bidir=0):
    chamferDist = ChamferDistance()
    parameters_smpl = OptimizationSMPL().cuda()
    parameters_smpl.pose = init['pose']
    parameters_smpl.beta = init['beta']
    parameters_smpl.trans = init['trans']
    parameters_smpl.scale = init['scale']
    
    lr = 2e-2
    
    optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
    iterations = 500
    ind_verts = np.arange(6890)
    pred_mesh_torch = torch.FloatTensor(pred_mesh).cuda()

    factor_beta_reg = 0.2

    for i in tqdm.tqdm(range(iterations),desc="Chamfer"):
        pose, beta, trans, scale = parameters_smpl.forward()
        #beta = beta*3
        vertices_smpl = (SMPL_model.forward(theta=pose, beta=beta, get_skin=True)[0][0] + trans)*scale
        distances = torch.abs(pred_mesh_torch - vertices_smpl)

        if bidir==0:
            d1 = torch.sqrt(chamferDist(torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), vertices_smpl.unsqueeze(0), False)).mean()
            d2 = torch.sqrt(chamferDist(vertices_smpl.unsqueeze(0), torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), False)).mean()

            loss = d1 + d2
        elif bidir==1: ## Partial
            loss = torch.sqrt(chamferDist(torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), vertices_smpl.unsqueeze(0), False)).mean()
        elif bidir==-1: ##Clutter
            loss = torch.sqrt(chamferDist(vertices_smpl.unsqueeze(0), torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), False)).mean()

        prior_loss = prior.forward(pose[:, 3:], beta)
        beta_loss = (beta**2).mean()
        loss = loss + prior_loss*0.00000001 + beta_loss*factor_beta_reg

        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()
        
        for param_group in optimizer_smpl.param_groups:
            param_group['lr'] = lr*(iterations-i)/iterations

    with torch.no_grad():
        pose, beta, trans, scale = parameters_smpl.forward()
        #beta = beta*3
        vertices_smpl = (SMPL_model.forward(theta=pose, beta=beta, get_skin=True)[0][0] + trans)*scale
        pred_mesh3 = vertices_smpl.cpu().data.numpy()
        params = {}
        params['loss'] = loss 
        params['beta'] = beta 
        params['pose'] = pose 
        params['trans'] = trans 
        params['scale'] = scale
    return pred_mesh3, params
                    
def get_match_LVD(s_src, s_tar, reg_src, reg_tar):
    # Returns for each point of s_src the match for s_tar
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(reg_src)
    distances, result_s = nbrs.kneighbors(s_src)


    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(s_tar)
    distances, result_t = nbrs.kneighbors(reg_tar[np.squeeze(result_s)])

    result = np.squeeze(result_t)
    return result


def vox_scan(scan, res, style='occ', grad=0,device='cuda', margin=0.8,center=True,scale=True):
    if style=='occ':
        voxel_src, mesh = voxelize(scan, res) #self.voxelize_scan(scan)
    else:
        voxel_src, mesh = voxelize_distance(scan, res)
        
        
    voxel_src = torch.FloatTensor(voxel_src)[None, None].to(device)
    
    # Encode efficiently to input to network:
    if grad and style=='occ_dist':
        voxel_src = torch.cat((torch.clamp(voxel_src, 0, 0.01)*100,
                                        torch.clamp(voxel_src, 0, 0.02)*50,
                                        torch.clamp(voxel_src, 0, 0.05)*20,
                                        torch.clamp(voxel_src, 0, 0.10)*20,
                                        torch.clamp(voxel_src, 0, 0.15)*15,
                                        torch.clamp(voxel_src, 0, 0.20)*10,
                                        
                                        torch.gradient(voxel_src,axis=2)[0]*grad,
                                        torch.gradient(voxel_src,axis=3)[0]*grad,
                                        torch.gradient(voxel_src,axis=4)[0]*grad,
                                        
                                        torch.clamp((torch.abs(torch.gradient(voxel_src,axis=2)[0]) + torch.abs(torch.gradient(voxel_src,axis=3)[0]) + torch.abs(torch.gradient(voxel_src,axis=4)[0])),-0.1,0.1)*grad,
                                        torch.clamp((torch.abs(torch.gradient(voxel_src,axis=2)[0]) + torch.abs(torch.gradient(voxel_src,axis=3)[0]) + torch.abs(torch.gradient(voxel_src,axis=4)[0])),-0.1,0.1)*grad,
                                        torch.clamp((torch.abs(torch.gradient(voxel_src,axis=2)[0]) + torch.abs(torch.gradient(voxel_src,axis=3)[0]) + torch.abs(torch.gradient(voxel_src,axis=4)[0])),-0.1,0.1)*grad,                                     
                                        
                                        voxel_src
                                        ), 1)        
    else:
        voxel_src = torch.cat((torch.clamp(voxel_src, 0, 0.01)*100,
                            torch.clamp(voxel_src, 0, 0.02)*50,
                            torch.clamp(voxel_src, 0, 0.05)*20,
                            torch.clamp(voxel_src, 0, 0.1)*20,
                            torch.clamp(voxel_src, 0, 0.15)*15,
                            torch.clamp(voxel_src, 0, 0.2)*10,
                            voxel_src
                            ), 1)
        voxel_src = torch.reshape(voxel_src,(1,7,res,res,res))
    
    return voxel_src, mesh


def fit_LVD(module, gt_points, voxel_src, iters=20, init=None):
    with torch.no_grad():
        if init is None:
            input_points = torch.zeros(1, gt_points, 3).cuda()
        else:
            input_points = init.cuda()
            
        _B = 1
        module.model(voxel_src)
        inds = np.arange(gt_points)
        for it in tqdm.tqdm(range(iters), desc="NF CONVERGENCE"):
            pred_dist = module.model.query(input_points)
            res = type(pred_dist) is tuple
            if res:
                pred_dist = pred_dist[0]
            pred_dist = pred_dist.reshape(_B, gt_points, 3, -1).permute(0, 1, 3, 2)
            input_points = - pred_dist[:, inds, inds] + input_points
        reg_src = input_points[0].cpu().data.numpy()
        
    return reg_src


def compute_curve(errors, thresholds):
    npoints = errors.shape[0]
    curve = np.zeros((len(thresholds)))
    for i in np.arange(0,len(thresholds)):
        curve[i] = 100*np.sum(errors <= thresholds[i])/ npoints;
    return curve

import itertools
def selfsup_module(module, voxel_src, input_points, gt_points):
    module.train()  

    paramets = itertools.chain(module.model.conv_1.parameters(), module.model.conv_1_1.parameters(), 
                    module.model.conv_2.parameters(), module.model.conv_2_1.parameters(), 
                    module.model.conv_3.parameters(), module.model.conv_3_1.parameters(), 
                    module.model.conv_4.parameters(), module.model.conv_4_1.parameters(), 
                    module.model.conv_5.parameters(), module.model.conv_5_1.parameters(), 
                    module.model.conv_6.parameters(), module.model.conv_6_1.parameters(), 
                    module.model.conv_7.parameters(), module.model.conv_7_1.parameters(), 
                    module.model.fc_0.parameters(), module.model.fc_1.parameters(), 
                    module.model.fc_2.parameters(), module.model.fc_3.parameters(), 
                    module.model.fc_4.parameters(), module.model.fc_out.parameters()
                    )
    
    optimizer = torch.optim.Adam(paramets, lr=0.001)
    input_points = torch.unsqueeze(torch.autograd.Variable(torch.Tensor(np.asarray(input_points)),requires_grad=False),0).cuda().detach()

    
    chamferDist = ChamferDistance()

    for i in np.arange(0,10):
        optimizer.zero_grad()
        module.model(voxel_src)
        
        pred_scan = module.model.self_sup()
        pred_scan = pred_scan.reshape(gt_points, 3)
        
        
        d1 = torch.sqrt(chamferDist(pred_scan.unsqueeze(0), input_points, False)).mean()
        d2 = torch.sqrt(chamferDist(input_points, pred_scan.unsqueeze(0), False)).mean()

        factor = max(1, int(input_points.shape[1] / 20000))
        
        input_points_res = input_points[:,1:input_points.shape[1]:factor,:]
        pred_dist = module.model.query(input_points_res)
        pred_dist = pred_dist.reshape(1, gt_points, 3, -1).permute(0, 1, 3, 2)
        v, _ = torch.min(torch.sum(pred_dist**2,axis=3),axis=1)
        
        loss = d1 + d2 + torch.sum(v)*1e-5

        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(i, loss.item()))

        
    module.eval()    
    return module 



import robust_laplacian
import trimesh 

class OptimizationOffsets(torch.nn.Module):
    def __init__(self,n_v=6890):
        super(OptimizationOffsets, self).__init__()
        self.offsets = torch.nn.Parameter(torch.zeros(n_v, 3).cuda())

    def forward(self):
        return self.offsets


def fit_plus_D(out_cham_s, SMPL_model, target, lambda_d1=0.01, lambda_lapl = 10000, lambda_reg = 5.0, iterations=1000,lr = 1e-4, subdiv=0):
    with torch.no_grad():
        faces = SMPL_model.faces
        L, M = robust_laplacian.mesh_laplacian(np.asarray(out_cham_s), np.asarray(faces))
        L = L.todense().astype(np.float32)
    smpld_vertices, params = fit_plus_D_sub(L, out_cham_s, faces , target, lambda_d1, lambda_lapl, lambda_reg,iterations,lr)
    p = {}
    p[0] = params    

    if subdiv:
        for j in range(subdiv):
            A_our = trimesh.Trimesh(smpld_vertices, faces)
            A_our = A_our.subdivide()
            
            out_cham_s = np.asarray(A_our.vertices)
            faces = np.asarray(A_our.faces)
            del L, M
            
            L, M = robust_laplacian.mesh_laplacian(out_cham_s, faces)
            L = L.todense().astype(np.float32)           
            smpld_vertices, _ = fit_plus_D_sub(L, out_cham_s, faces , target, lambda_d1, lambda_lapl, lambda_reg,iterations,lr)
            #p[j+1] = params_2
    
    return smpld_vertices, faces, p    
        


def fit_plus_D_sub(L, out_cham_s, faces, target, lambda_d1=0.01, lambda_lapl = 10000, lambda_reg = 5.0,iterations=1000,lr = 1e-4):

    with torch.no_grad():
        chamferDist = ChamferDistance()
        L = torch.tensor(L,dtype=torch.float32,requires_grad=False).cuda()
        L.requires_grad = False
        
        # L, M = robust_laplacian.mesh_laplacian(out_cham_s, np.asarray(SMPL_model.faces))
        # L = torch.FloatTensor(L.todense()).cuda()
        # init_smooth = L @ torch.FloatTensor(out_cham_s).cuda()
        
        vertices_smpl_fit = torch.tensor(out_cham_s,dtype=torch.float32,requires_grad=False).cuda()
        parameters_offsets = OptimizationOffsets(out_cham_s.shape[0]).cuda()
        
        
        optimizer_offsets = torch.optim.Adam(parameters_offsets.parameters(), lr=lr)
        offsets = parameters_offsets.forward()
        
        # Starting SMPL
        vertices_smpl = vertices_smpl_fit + offsets
        # Starting Smoothness
        init_smooth = L @ vertices_smpl

    for i in tqdm.tqdm(range(iterations), desc="FIT SMPL+D"):
        offsets = parameters_offsets.forward()
        vertices_smpl = vertices_smpl_fit + offsets

        d1 = torch.sqrt(chamferDist(torch.FloatTensor(target).cuda().unsqueeze(0), vertices_smpl.unsqueeze(0), False)).mean()
        d2 = torch.sqrt(chamferDist(vertices_smpl.unsqueeze(0), torch.FloatTensor(target).cuda().unsqueeze(0), False)).mean()

        lapl = ((L @ vertices_smpl - init_smooth)**2).mean()
        beta_loss = (offsets**2).mean()

        loss = d1*lambda_d1 + d2 + lapl*lambda_lapl + beta_loss*lambda_reg

        optimizer_offsets.zero_grad()
        loss.backward()
        optimizer_offsets.step()

    for param_group in optimizer_offsets.param_groups:
        param_group['lr'] = lr*(iterations-i)/iterations
        
    with torch.no_grad():
        offsets = parameters_offsets.forward()
        smpld_vertices = (vertices_smpl_fit + offsets).cpu().data.numpy()

    params = {}
    params['loss_d1'] = d1.detach().item() 
    params['loss_d2'] = d2.detach().item()
    params['offsets'] = np.asarray(offsets.detach().cpu())

    return smpld_vertices, params


def procrustes(X, Y, scaling=True, reflection='best'):

    """

    A port of MATLAB's `procrustes` function to Numpy.



    Procrustes analysis determines a linear transformation (translation,

    reflection, orthogonal rotation and scaling) of the points in Y to best

    conform them to the points in matrix X, using the sum of squared errors

    as the goodness of fit criterion.



        d, Z, [tform] = procrustes(X, Y)



    Inputs:

    ------------

    X, Y    

        matrices of target and input coordinates. they must have equal

        numbers of  points (rows), but Y may have fewer dimensions

        (columns) than X.



    scaling 

        if False, the scaling component of the transformation is forced

        to 1



    reflection

        if 'best' (default), the transformation solution may or may not

        include a reflection component, depending on which fits the data

        best. setting reflection to True or False forces a solution with

        reflection or no reflection respectively.



    Outputs

    ------------

    d       

        the residual sum of squared errors, normalized according to a

        measure of the scale of X, ((X - X.mean(0))**2).sum()



    Z

        the matrix of transformed Y-values



    tform   

        a dict specifying the rotation, translation and scaling that

        maps X --> Y



    """



    n,m = X.shape

    ny,my = Y.shape



    muX = X.mean(0)

    muY = Y.mean(0)



    X0 = X - muX

    Y0 = Y - muY



    ssX = (X0**2.).sum()

    ssY = (Y0**2.).sum()



    # centred Frobenius norm

    normX = np.sqrt(ssX)

    normY = np.sqrt(ssY)



    # scale to equal (unit) norm

    X0 /= normX

    Y0 /= normY



    if my < m:

        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)



    # optimum rotation matrix of Y

    A = np.dot(X0.T, Y0)

    U,s,Vt = np.linalg.svd(A,full_matrices=False)

    V = Vt.T

    T = np.dot(V, U.T)



    if reflection != 'best':



        # does the current solution use a reflection?

        have_reflection = np.linalg.det(T) < 0



        # if that's not what was specified, force another reflection

        if reflection != have_reflection:

            V[:,-1] *= -1

            s[-1] *= -1

            T = np.dot(V, U.T)



    traceTA = s.sum()



    if scaling:



        # optimum scaling of Y

        b = traceTA * normX / normY



        # standarised distance between X and b*Y*T + c

        d = 1 - traceTA**2



        # transformed coords

        Z = normX*traceTA*np.dot(Y0, T) + muX



    else:

        b = 1

        d = 1 + ssY/ssX - 2 * traceTA * normY / normX

        Z = normY*np.dot(Y0, T) + muX



    # transformation matrix

    if my < m:

        T = T[:my,:]

    c = muX - b*np.dot(muY, T)

    

    #transformation values 

    tform = {'rotation':T, 'scale':b, 'translation':c}

   

    return d, Z, tform

