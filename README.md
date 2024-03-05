<p align="center">
  <h1 align="center"> Neural ICP Fields for 3D Human Registration at Scale
 </h1>
 <p align="center">
    <a href="https://riccardomarin.github.io/"><strong>Riccardo Marin</strong></a>
    ·
    <a href="https://www.iri.upc.edu/people/ecorona/"><strong>Enric Corona</strong></a>
    .
    <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html"><strong>Gerard Pons-Moll</strong></a>
  </p>
  <h2 align="center"> </h2>
  <div align="center">
    <img src="assets/myteaser3.png" alt="teaser" width="100%">
  </div>
</p> 

 [[arXiv]](https://arxiv.org/abs/2312.14024)


## Getting Started
Clone the repo 
  ```bash
  git clone https://github.com/riccardomarin/INLoVD.git INLoVD
  cd INLoVD
  ```  

Create the environment 
 ```
conda create -n inlovd python=3.8.13
conda activate inlovd
  ```

Run the installation script (it also contains checkpoint download)
 ```
./install.sh
  ```

Finally, you need to download the smplh model and place it in the ``support_data`` folder. The correct file structure is:
```
support_data
  |__ body_models
         |__smplh
             |__neutral
                 |__model.npz
```
You are ready to start!

## Inference 
To use INLoVD and fit all the scans into the ``demo`` folder, you can run the following command:

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py
```

We also provide a streamlit demo to run INLoVD on single shapes using a GUI.
```
PYTHONPATH=. streamlit run ./src/lvd_templ/evaluation/stream_demo.py
```

## Change Settings in ``evaluation_benchmark.py``
Depending on your use case, you may specify different paremeters for the INLoVD pipeline. This can be easily done my command line. For example:


```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.cham_bidir=-1
```

will run a unidirectional chamfer distance refinement (-1: input with oultiers; 1: Partial input).

If you want to characterize a run (and avoid overwrite), you can specify a tag:

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.tag='greedy' core.lr_ss=0.01
```

A complete list of available parameters can be found in ``conf_test/default.yaml``.


<!-- 
<img src='./assets/myteaser3.png' width=800> -->
