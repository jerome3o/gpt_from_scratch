# Setting up a deep learning rig

## Hardware

TODO: copy from sheet

## Software

* os - ubuntu server 20.04LTS

### Setting up ROCm

Install the `amdgpu-install` script

```sh
sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/5.4.3/ubuntu/focal/amdgpu-install_5.4.50403-1_all.deb  
sudo apt-get install ./amdgpu-install_5.4.50403-1_all.deb
```

Install the latest ROCm version

```sh 
sudo amdgpu-install --usecase=rocm --rocmrelease=5.4.3
```

Reboot

```sh
sudo reboot
```

Add your user to `video` and `render` groups

```sh
sudo usermod -aG video jerome
sudo usermod -aG render jerome
```

### Testing ROCm

```sh
rocminfo
# TODO: show outputs
```

### Testing PyTorch

```sh
# make a dir for testing ROCm stuff
mkdir ROCm && cd ROCm
python3 -m venv venv # might need to install python3.X-venv

# pip install pytorch

python -c "import torch; torch.cuda.is_available()"
```
