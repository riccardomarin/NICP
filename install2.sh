### Create Environment
# conda create -n inlovd_git python=3.8.13
# conda activate inlovd3_git

# Download checkpoint
wget oshi.at/hQWV 1ljjfnbx.zip
unzip hQWV.1 -d ./storage/
rm hQWV.1

### Install Pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

### External Dependency: Human Body Prior
git clone https://github.com/nghorbani/human_body_prior.git human_body_prior_git
cd human_body_prior_git/
pip install -r requirements.txt
python setup.py develop
cd ..
cp -r ./human_body_prior_git/src/* .

### External Dependency: IF-NET voxel preprocessings
pip install cython
git clone https://github.com/jchibane/if-net if-net 
mkdir ./data/preprocess_voxels
cp -r ./if-net/data_processing/* ./data/preprocess_voxels
rm -rf if-net 

cd ./data/preprocess_voxels/libmesh
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../../..

### External Dependency: LVD Utils
git clone https://github.com/enriccorona/LVD.git lvd
cp -r lvd/utils/ utils_cop
rm -rf lvd
# in SMPL.py
# update file locations for neutral_smpl_with_cocoplus_reg.txt 
sed -i -e 's/np.float/float/g' ./utils_cop/SMPL.py
sed -i -e 's/utils\/shapedirs_300.npy/utils_cop\/shapedirs_300.npy/g' ./utils_cop/SMPL.py   

### Install needed libraries and current project
pip install hydra-core==1.2.0 pytorch-lightning==1.5.10 open3d==0.15.2 trimesh==3.13.0 opencv-python==4.6.0.66 scikit-image==0.19.3 robust-laplacian==0.2.4 plotly==5.10.0

# pip install -e .
pip install scikit-learn==1.1.2

