# Temporal copying and local hallucination for video inpainting
[![](https://img.shields.io/badge/publication-UPC%20Commons-red)](https://upcommons.upc.edu/handle/2117/334882)
[![](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![](https://www.codefactor.io/repository/github/davidalvarezdlt/master_thesis/badge)](https://www.codefactor.io/repository/github/davidalvarezdlt/master_thesis)
[![](https://img.shields.io/github/license/davidalvarezdlt/master_thesis)](https://github.com/davidalvarezdlt/master_thesis/blob/main/LICENSE)

This repository contains the implementation of my master's thesis ["Temporal copying and local hallucination for video inpainting"](https://upcommons.upc.edu/handle/2117/334882).
The code has been built using PyTorch Lightning, read its documentation to get a
complete overview of how this repository is structured.

**Disclaimer**: The version published here might contain small differences with
the thesis because of the refactoring.

## About the data

The thesis uses three different datasets: GOT-10k for the background sequences,
YouTube-VOS for realistic mask shapes and DAVIS to test the models with real
masked sequences. Some pre-processing steps, which are not published in this
repository, have been applied to the data. You can download the exact datasets
used in the paper from [this link](https://www.kaggle.com/davidalvarezdlt/master-thesis).

The first step is to clone this repository, install its dependencies and
other required system packages:

```
git clone https://github.com/davidalvarezdlt/master_thesis.git
cd master_thesis
pip install -r requirements.txt

apt-get update
apt-get install libturbojpeg ffmpeg libsm6 libxext6
```

Unzip the file downloaded from the previous link inside ``./data``. The
resulting folder structure should look like this:

```
master_thesis/
    data/
        DAVIS-2017/
        GOT10k/
        YouTubeVOS/
    lightning_logs/
    master_thesis/
    .gitignore
    .pre-commit-config.yaml
    LICENSE
    README.md
    requirements.txt
```

## Training the Dense Flow Prediction Network (DFPN) model

In short, you can train the model by calling:

```
python -m master_thesis
```

You can modify the default parameters of the code by using CLI parameters. Get a
complete list of the available parameters by calling:

```
python -m master_thesis --help
```

For instance, if we want to train the model using 2 frames, with a batch size of
8 and using one GPUs, we would call:

```
python -m master_thesis --frames_n 2 --batch_size 8 --gpus 1
```

Every time you train the model, a new folder inside ``./lightning_logs`` will be
created. Each folder represents a different version of the model, containing its
checkpoints and auxiliary files.

## Training the Copy-and-Hallucinate Network (CHN) model

In this case, you will need to specify that you want to train the CHN model. To
do so:

```
python -m master_thesis --chn --chn_aligner <chn_aligner> --chn_aligner_checkpoint <chn_aligner_checkpoint>
```

Where ``--chn_aligner`` is the model used to align the frames (either ``cpn`` or
``dfpn``) and ``--chn_aligner_checkpoint`` is the path to its checkpoint.

You can download the checkpoint of the CPN from its [original repository](https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting) (file named ``weight.pth``).

## Testing the Dense Flow Prediction Network (DFPN) model

You can align samples from the test split and store them in TensorBoard by
calling:

```
python -m samplernn_pase --test --test_checkpoint <test_checkpoint>
```

Where ``--test_checkpoint`` is a valid path to the model checkpoint that should
be used.

## Testing the Copy-and-Hallucinate Network (CHN) model

You can inpaint test sequences (they will be stored in a folder) using the three
algorithms by calling:

```
python -m master_thesis --chn --chn_aligner <chn_aligner> --chn_aligner_checkpoint <chn_aligner_checkpoint> --test --test_checkpoint <test_checkpoint>
```

Notice that now the value of ``--test_checkpoint`` must be a valid path to a
CHN checkpoint, while ``--chn_aligner_checkpoint`` might be the path to a
checkpoint of either CPN or DFPN.

## Citation

If you find this thesis useful, please use the following citation:

```
@thesis{Alvarez2020,
    type = {Master's Thesis},
    author = {David Álvarez de la Torre},
    title = {Temporal copying and local hallucination for video onpainting},
    school = {ETH Zürich},
    year = 2020,
}
```