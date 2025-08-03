# Master's Thesis Repository: Deep Learning Based Sequence Models for Music Generation

## Overview

Welcome to the repository for my Master's thesis titled "Deep Learning Based Sequence Models for Music Generation." This research explores the conditioned music generating capabilities of the Transformer, Mamba, and xLSTM architectures on symbolic music. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)

## Introduction

Recent advancements in deep learning, particularly in sequence modeling architectures such as xLSTM and Mamba, have allowed for improved context-windows and claims to outperform the previous Transformer model in large language models of similar size. On the subject of music generation, the overarching theme of the music can spans many thousands of tokens, depending on the representation of the music. It is therefore of great interest to test these new models to find their potential in generating high-quality, conditioned, polyphonic compositions from discrete symbolic representation. Transformer-based models faced limitations due to the quadratic scaling of architecture size with context window length, but with the linear scaling in architectural size with context window length of the xLSTM and Mamba model.

## Installation

To set up the environment for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/thorGabe123/Deep-Learning-Based-Sequence-Models-for-Music-Generation.git
   cd Deep-Learning-Based-Sequence-Models-for-Music-Generation

## Usage

An example of training the model using 2 GPUs
```
cd Deep-Learning-Based-Sequence-Models-for-Music-Generation
torchrun --nproc_per_node=2 train_parallel.py --model mamba
```

An example of generating 2000 token samples from the Mamba model
```
cd Deep-Learning-Based-Sequence-Models-for-Music-Generation/scripts
python generate_midi_combined.py --length 2000 --mamba True --composers "Mozart, Bach, Beethoven, Chopin, Liszt"
```

## Project Structure

```bash
├── configs          
│   ├── common            # Common configuration settings
│   │   ├── init.py
│   │   └── config.yaml
│   ├── mamba             # Mamba model configuration (Not used)
│   │   ├── init.py
│   │   └── config.yaml
│   ├── paths             # Path configurations
│   │   ├── init.py
│   │   └── config.yaml
│   ├── transformer       # Transformer model configuration
│   │   ├── init.py
│   │   └── config.yaml
│   └── xlstm             # XLSTM model configuration (Not used)
│       ├── init.py
│       └── config.yaml
├── models
│   ├── init.py
│   ├── classifier        # Classifier model code
│   │   ├── init.py
│   │   └── model.py
│   ├── mamba             # Mamba model code
│   │   ├── init.py
│   │   └── mamba.py
│   ├── transformer       # Transformer model code
│   │   ├── init.py
│   │   └── model_transformer.py
│   └── xlstm             # XLSTM model code
│       ├── init.py
│       └── xlstm_model.py
├── note.py               # Musical Note class
├── pretrained            # Pretrained model directories (Models were too large to include in GitHub Repository)
│   ├── classifier
│   ├── mamba
│   ├── transformer
│   └── xlstm
├── processing            # Data processing scripts
│   ├── init.py
│   ├── dataset.py        # Dataset handling
│   └── processing.py     # Data processing functions
├── requirements.txt      # Python dependencies
├── scripts               # Folder containing testing scripts, visualization scipts, midi generating scripts, etc.
├── train.py              # Training script for models
├── train_parallel.py     # Training script for multiple GPU training
├── train_classifier.py   # Training script for classifier
└── samples               # Audio of composer conditioned music used for the qualitative testing
    ├── Beethoven
    ├── Brahms
    ├── Chopin
    ├── Debussy
    ├── Liszt
    ├── Mozart
    ├── Vivaldi
    ├── Wagner
    └── Extra             # Audio or midi samples not used during testing 

```

Our project utilizes the following MIDI music datasets:

- [**MIDI Classical Music** (HuggingFace)](https://huggingface.co/datasets/drengskapur/midi-classical-music):  
  A collection of classical music MIDI files.

- [**Lakh MIDI Clean** (Kaggle)](https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean):  
  A large-scale, cleaned version of the Lakh MIDI dataset containing a wide range of genres and compositions.