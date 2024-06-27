# LoVD Training
This folder contains the code to train LoVD. 

# 1 - Data preprocessing
To preprocess the data, enter the folder ``data_preprocessing`` and follow the instructions.

# 2 - Set up the environment
Run the following commands to create the environment

```
conda create -n lovd_t python=3.8.13
conda activate lovd_t
pip install -e .

# Human Body Prior
git clone https://github.com/nghorbani/human_body_prior.git human_body_prior_git
cd human_body_prior_git/
pip install -r requirements.txt
python setup.py develop
cd ..
cp -r ./human_body_prior_git/src/* .
rm -rf human_body_prior_git/

git clone https://github.com/enriccorona/LVD.git lvd
cp -r lvd/utils/ utils_cop
rm -rf lvd
# in SMPL.py
# update file locations for neutral_smpl_with_cocoplus_reg.txt 
sed -i -e 's/np.float/float/g' ./utils_cop/SMPL.py
sed -i -e 's/utils\/shapedirs_300.npy/utils_cop\/shapedirs_300.npy/g' ./utils_cop/SMPL.py   

pip install cython
git clone https://github.com/jchibane/if-net if-net 
mkdir ./data/
mkdir ./data/preprocess_voxels
cp -r ./if-net/data_processing/* ./data/preprocess_voxels
rm -rf if-net 

cd ./data/preprocess_voxels/libmesh
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../../..


pip install hydra-core==1.2.0
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pytorch_lightning==1.5.10 nn-template-core==0.1.1 open3d==0.15.2 robust-laplacian==0.2.4 trimesh==3.13.0 plotly==5.10.0 scikit-learn==1.1.2
```

Similar for inference, you need to download the neutral SMPL+H model and save it in support data:
```
support_data
|_ body_models
   |_ smplh
      |_ neutral
         |_ model.npz
```

# 2 - Lunch training
```
PYTHONPATH=. python ./src/lvd_templ/run.py core.tags=['TEST']
```