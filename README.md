# PerFRDiff: Personalised Weight Editing for Multiple Appropriate Facial Reaction Generation
This repository contains a pytorch implementation of "PerFRDiff: Personalised Weight Editing for Multiple Appropriate Facial Reaction Generation".


## 👨‍🏫 Main Sections:
[//]: # (- [Overview]&#40;#overview&#41;)
[//]: # (- [Installation]&#40;#installation&#41;)
[//]: # (- [Dataset]&#40;#dataset&#41;)
[//]: # (- [Usage]&#40;#usage&#41;)

<details>
<summary><b>🛠️ Dependency Installation</b></summary>
<p>

We provide detailed instructions for setting up the environment using conda. First, create and activate a new environment:
``` shell
conda create -n react python=3.10
conda activate react
```

### 1. Install PyTorch
First, check your CUDA version:
``` shell
nvidia-smi
```
Visit [Pytorch official website](https://pytorch.org/) to get the appropriate installation command. For example:
``` shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install PyTorch3D Dependencies
Install the following dependencies:
``` shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
For CUDA versions older than 11.7, you will need to install the CUB library. There are two installation options:

Option A: Using conda (Recommended)
``` shell
conda install -c bottler nvidiacub
```
Option B: Manual installation
1. Download the CUB library from NVIDIA CUB Releases.
2. Unpack it to a folder of your choice. For example, on Linux/Mac:
``` shell
cd ~
mkdir CUB
curl -LO https://github.com/NVIDIA/cub/archive/2.1.0.tar.gz
tar xzf 2.1.0.tar.gz
```
[//]: # (Define the environment variable CUB_HOME in `~/.bashrc` before building and point it to the directory that contains `CMakeLists.txt` for CUB.)
3. Define the environment variable CUB_HOME in `~/.bashrc`. This variable should point to the directory that contains `CMakeLists.txt` for CUB. Add this line to your `~/.bashrc`:
``` shell
export CUB_HOME=~/CUB/cub-2.1.0
```

[//]: # (Install jupyter-notebook and make the environment `react` available to jupyter-notebook by running)
To enable Jupyter notebook support, install Jupyter and register the environment:
``` shell
conda install jupyter
python -m ipykernel install --user --name=react
```

### 3. Install PyTorch3D
First, verify your CUDA version in Python:
``` shell
import torch
torch.version.cuda
```
[//]: # (Download `pytorch3d` file based on the version of python, cuda and pytorch from https://anaconda.org/pytorch3d/pytorch3d/files. For example, to install for Python 3.8, PyTorch 1.12.1 and CUDA 11.6, select the below file to download)
Download the appropriate `PyTorch3D` package from [Anaconda](https://anaconda.org/pytorch3d/pytorch3d/files) based on your Python, CUDA, and PyTorch versions. For example, for Python 3.10, CUDA 11.6, and PyTorch 1.12.0:

[//]: # (Finally install `pytorch3d` via the downloaded `.tar.bz2` file via conda)
``` shell
# linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
conda install linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
```

### 4. Install Additional Dependencies
[//]: # (pip install omegaconf scikit-video pandas soundfile av decord tensorboard numpy tslearn scikit-image matplotlib imageio plotly opencv-python librosa einops)
Install all remaining dependencies specified in requirements.txt:
``` shell
pip install -r requirements.txt
```

</p>
</details>

<details>
<summary><b>📊 Dataset</b></summary>
<p>

Our work is built upon the REACT 2024 Multimodal Challenge Dataset, which leverages two well-established dyadic interaction datasets: [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780) and [RECOLA](https://ieeexplore.ieee.org/document/6553805). The dataset can be accessed through the official [REACT 2024 Challenge Homepage](https://sites.google.com/cam.ac.uk/react2024).

After downloading the dataset, please rename your downloaded folder to `data` and place it in the root directory of this project.

### Data Structure
Example directory structure:
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1.mp4
                   ├── ....
           ├── ....
       ├── RECOLA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P1
               ├── P2
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
```
</p>
</details>

<details>
<summary><b>📖 Usage</b></summary>
<p>

### Pre-trained Models
This project provides several pre-trained models, such as:
* Generic Appropriate Facial Reaction Generator (GAFRG)
* Personalized Weight Shifts Generation (PWSG) Block
* Personalized Style Space Learning (PSSL) Block

You can access and download all the available pre-trained models from the following [Google Drive link](https://drive.google.com/file/d/1Drdq3WnQjuOM9GxptC3UsTn_JSsn8_M-/view?usp=sharing). After downloading, please unzip the file and place the `checkpoints` folder into the root directory of this project.

### External Dependencies
Our framework leverages two key external tools:
- [FaceVerse](https://github.com/LizhenWangT/FaceVerse) for extraction of 3DMM coefficients
- [PIRender](https://github.com/RenYurui/PIRender) (3D-to-2D tool) for facial reaction frame rendering

For convenience, we have compiled all necessary model files into a single package, available at [Google Drive link](https://drive.google.com/file/d/14Z1vFbu4Amren8TjS_PEIZZVIqWdikm5/view?usp=sharing). After downloading, please extract the `external` folder and place it in the root directory of this project. This package includes:

- FaceVerse model (Version 2) and auxiliary files (mean_face, std_face, and reference_full)
- Well-trained PIRender model

### Training
``` bash
# Training GAFRG for multiple appropriate facial reaction generation
python train_diffusion.py --mode train --writer True --config diffusion_model.yaml
```

``` bash
# Training Personalized GAFRG (with Weight Editing) for multiple appropriate facial reaction generation
python train_rewrite_weight.py --mode train --writer True --config rewrite_weight.yaml
```

### Inference
``` bash
# Inference using GAFRG for multiple appropriate facial reaction generation
python evaluate_diffusion.py --mode test --config diffusion_model.yaml
```

``` bash
# Inference using Personalised GAFRG (with Weight Editing) for multiple appropriate facial reaction generation
python evaluate_rewrite_weight.py --mode test --config rewrite_weight.yaml
```

</p>
</details>


## 📽 Visualization of Facial Reactions

### Qualitative Results

[//]: # (Qualitative comparison between our method and existing baselines.)
[//]: # (![Comparison]&#40;docs/figures/comparison.png&#41;)

<details>
<summary>Qualitative comparison between our method and existing baselines (Click to expand) </summary>
<p>

The presentation figures are maintained outside this code-only repository.

</p>
</details>

### Dynamic Visualization

https://github.com/user-attachments/assets/5799f032-3fbf-4f41-bd29-6293c6f4e151


## 🤝 Acknowledgement
We extend our sincere gratitude to the following open-source projects:
- [FaceVerse](https://github.com/LizhenWangT/FaceVerse)
- [PIRender](https://github.com/RenYurui/PIRender)


## 🖊️ Citation
```
@inproceedings{zhu2024perfrdiff,
  title={Perfrdiff: Personalised weight editing for multiple appropriate facial reaction generation},
  author={Zhu, Hengde and Kong, Xiangyu and Xie, Weicheng and Huang, Xin and Shen, Linlin and Liu, Lu and Gunes, Hatice and Song, Siyang},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9495--9504},
  year={2024}
}
```

## Integrated audio baseline

The `team1/integrated-baseline` branch adds the multimodal audio-response baseline
on top of the original PerFRDiff code. The original training/evaluation scripts
are preserved; the new entry points are `train.py` and `infer.py`.

### Environment

Use the verified React CUDA environment specification:

```bash
conda env create -f requirements/environment.react.yml
conda activate react
```

The legacy `requirements.txt` is retained for the original PerFRDiff scripts.

### Data split and tests

Build or refresh the conversation-level split manifest before training:

```bash
python scripts/build_split_manifest.py --output artifacts/splits/conv_id_seed42.json
python -m unittest discover -s tests -v
```

The manifest keeps conversation IDs disjoint between train, validation, and test
sets to avoid leakage. Dataset and checkpoint paths are supplied through the
training/inference arguments; large datasets, checkpoints, and generated outputs
are intentionally not committed to Git.

### Training and inference

Inspect all available options with:

```bash
python train.py --help
python infer.py --help
```

For the full verified CUDA setup, use the commands documented in the team
integration notes outside this code-only repository. A missing inference
checkpoint fails explicitly instead of producing an untrained random sample.
