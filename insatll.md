# 🚀 FlowRAM: Installation Guide

To set up the **FlowRAM** environment, first ensure that any existing environment named `flowram` is removed using:

```bash
conda env remove -n flowram
```

Update Conda to the latest version to avoid compatibility issues:

```bash
conda update conda
```

Now create the environment from the provided `environment.yaml` file and activate it:

```bash
conda env create -f environment.yaml
conda activate flowram
```

Install the Diffusers library with PyTorch support:

```bash
pip install 'diffusers[torch]'
```

Clone the **PointMamba** repository and install its Python dependencies:

```bash
git clone https://github.com/LMD0311/PointMamba.git
cd ./FlowRAM/PointMamba
pip install -r requirements.txt
```

Some modules require compilation. Compile the **Chamfer Distance** extension:

```bash
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../..
```

Compile the **Earth Mover’s Distance (EMD)** extension:

```bash
cd ./extensions/emd
python setup.py install --user
cd ../..
```

Next, install additional dependencies. For **PointNet++** operations, run:

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

Install the **KNN\_CUDA** prebuilt wheel:

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Finally, install **Causal Convolution** and **Mamba-SSM**:

```bash
pip install causal-conv1d==1.0.0
pip install mamba-ssm==1.1.1
```

To verify the installation, activate the environment and run:

```bash
conda activate flowram
python -c "import torch; import diffusers; print('Setup successful!')"
```

This completes the environment setup. You are now ready to use **FlowRAM**.

