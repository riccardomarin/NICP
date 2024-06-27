import trimesh 
from sklearn.neighbors import NearestNeighbors
from lvd_templ.paths import path_FAUST_train_reg, path_FAUST_train_scans, path_challenge_pairs
import numpy as np
import tqdm
import hydra
from nn_core.common import PROJECT_ROOT
from hydra import compose, initialize
import omegaconf
from omegaconf import OmegaConf
from omegaconf import DictConfig
from lvd_templ.paths import  chk_pts, home_dir, output_dir
import os 

def compute_curve(errors, thresholds):
    npoints = errors.shape[0]
    curve = np.zeros((len(thresholds)))
    for i in np.arange(0,len(thresholds)):
        curve[i] = 100*np.sum(errors <= thresholds[i])/ npoints;
    return curve

def get_match_reg(s_src, s_tar, reg_src, reg_tar):
    # Returns for each point of s_src the match for s_tar
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(reg_src)
    distances, result_s = nbrs.kneighbors(s_src)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(s_tar)
    distances, result_t = nbrs.kneighbors(reg_tar[np.squeeze(result_s)])

    result = np.squeeze(result_t)
    return result


def run(cfg: DictConfig) -> str:
    os.chdir(PROJECT_ROOT)
    
    method = cfg['core'].evaluate
    challenge_name = cfg['core'].challenge.name
    
    if challenge_name == 'FAUST_train_reg':
        reg = 1
    else:
        reg = 0

    pairs = path_challenge_pairs
    file1 = open(pairs, 'r')
    Lines = file1.read().splitlines()
    x = [Lines[i].split('_') for i in range(len(Lines))]

    regs  = path_FAUST_train_reg
    scans = path_FAUST_train_scans

    scan_parser = lambda x: 'tr_scan_' + x.zfill(3)
    reg_parser = lambda x: 'tr_reg_' + x.zfill(3)

    errs = []
    
    # For every pair
    for p in tqdm.tqdm(x):
        # Load the matching
        
        if reg:
            matches = np.loadtxt('./output/FAUST_train_reg/' + method + '/' + p[0] + '_' + p[1] +'.txt')
        else:
            matches = np.loadtxt('./output/FAUST_train_scans/'      + method + '/' + p[0] + '_' + p[1] +'.txt')
            
            
        name_scan_1 = scans + scan_parser(p[0]) + '.ply'
        name_reg_1 = regs + reg_parser(p[0])+ '.ply'
        name_scan_2 = scans + scan_parser(p[1])+ '.ply'
        name_reg_2 = regs + reg_parser(p[1])+ '.ply'
        
        # Load the original shapes and obtain GT
        # if shapes have same triangulation, just use the natural vertices' order
        # otherwise use the GT registrations
        
        if not(reg):
            scan_1 = trimesh.load(name_scan_1, process=False, maintain_order=True)
            scan_2 = trimesh.load(name_scan_2, process=False, maintain_order=True)
            scan_1_v = np.asarray(scan_1.vertices)
            scan_2_v = np.asarray(scan_2.vertices)
            
        reg_1 = trimesh.load(name_reg_1, process=False, maintain_order=True)
        reg_2 = trimesh.load(name_reg_2, process=False, maintain_order=True)
        reg_1_v = np.asarray(reg_1.vertices)
        reg_2_v = np.asarray(reg_2.vertices)

        if reg:
            gt_match = np.arange(0,6890)#get_match_reg(scan_1_v, scan_2_v, reg_1_v, reg_2_v)
        else:
            gt_match = get_match_reg(scan_1_v, scan_2_v, reg_1_v, reg_2_v)
            
        if reg:    
            err = np.sqrt(np.sum((reg_2_v[matches.astype(np.int32)] - reg_2_v[gt_match.astype(np.int32)])**2,1))
        else:
            err = np.sqrt(np.sum((scan_2_v[matches.astype(np.int32)] - scan_2_v[gt_match.astype(np.int32)])**2,1))
            
        errs.append(err)
        
        tr = np.arange(0,0.5,0.01);  
        c  = compute_curve(np.hstack(errs).flatten(),tr);
        full_err = np.mean(np.hstack(errs).flatten())
    
    # Save errors, curves, and mean error in a dictionary
    dd = {}
    dd['tr'] = tr 
    dd['c'] = c
    dd['full_err'] = full_err
    import scipy.io as sio
    sio.savemat('./output/' + method + '_isreg_' + str(reg) + '.mat',dd)
    print("ERR " + method + " : " + str(full_err))


####
@hydra.main(config_path=str(PROJECT_ROOT / "conf_test"), config_name="default_err")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
    


