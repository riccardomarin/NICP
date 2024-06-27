# LoVD Training - Data Preprocessing
Huge thanks to [Ilya Petrov](https://github.com/ptrvilya) for helping with code refactoring.

This contains the code to preprocess AMASS. 
### Create the environment
```bash
conda create -n nficp_preproc python=3.8.13
conda activate nficp_preproc
```

### Install AMASS & Human Body Prior

```bash
AMASS_REPO="absolute path to clone the repo into"
HUMAN_BODY_PRIOR_REPO="absolute path to clone the repo into"

# AMASS
git clone https://github.com/nghorbani/amass.git ${AMASS_REPO}
cd ${AMASS_REPO}
pip install -r requirements.txt
python setup.py develop

# Human Body Prior
git clone https://github.com/nghorbani/human_body_prior.git ${HUMAN_BODY_PRIOR_REPO}
cd ${HUMAN_BODY_PRIOR_REPO}
pip install -r requirements.txt

# Other requirements
pip install tables trimesh scikit-image h5py tomlkit
```
Update `AMASS_REPO` and `HUMAN_BODY_PRIOR_REPO` in `config.toml` accordingly.
Download and extract AMASS into `AMASS_DATA` in `config.toml`.


### Download SMPL models
Extended SMPL+H model from MANO [website](https://mano.is.tue.mpg.de/download.php).
Path to the model (`SMPL_DIR` in `config.toml`):
```bash
${AMASS_REPO}/support_data/body_models/smplh/neutral/model.npz'
```


### Run preprocessing and voxelization
Check that the name of the dataset in the ``prepare_data_corr.py`` script are consistent with the folder names:
```
        'vald': ['HumanEva', 'HDM05', 'SFU', 'MoSh'],
        'test': ['Transitions', 'SSM'],
        'train': ['CMU', 'PosePrior', 'TotalCapture', 'EyesJapanDataset', 'KIT', 'BML', 'EKUT', 'TCDHands']
```
Then, you can run the data processing with:
```bash
python prepare_data_corr.py AUG_1_1
python preprocess_lib_vox_pt.py -e AUG_1_1 -r 64 -d vald train test -i /mnt/sdb/out_AMASS -j 4
```

If the process is successful, in the ``stage_III`` folder you should have this structure:

```
train
|_ifnet_indi
  |_occ_dist        (126.158 files)
  |_verts_occ_dist  (126.158 files)

vald
|_ifnet_indi
  |_occ_dist        (10.374 files)
  |_verts_occ_dist  (10.374 files)

test
|_ifnet_indi
  |_occ_dist        (867 files)
  |_verts_occ_dist  (867 files)

```


### Preprocess on your own data
Running the process on your data should require minor modifications since the processing is mainly to obtain the 64x64x64 voxelization of the training meshes. In the folder ``CAPE_example``, you will find two scripts (not runnable) that give an idea of the operations needed for that.