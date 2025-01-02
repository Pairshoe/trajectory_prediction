# Trajectory Prediction

This repository contains code for trajectory prediction using various models and datasets. The project is implemented in PyTorch and PyTorch Lightning.

## Requirements

The required packages are listed in the `requirements.txt` file.

## Installation

To install the required packages, run:

```sh
pip install -r requirements.txt
```

## Execution

### Training

To train the model, run the `train.py` script:

```python
python scripts/train.py
```

### Generating and Visualizing Trajectories

To generate and visualize trajectories, run the `generate.py` script:

```python
python scripts/generate.py
```

### Ablation Study

To perform an ablation study, run the `ablation.py` script:

```python
python scripts/ablation.py
```

## Project Components

### Models
* `SceneTransformer`: Main model module implemented in `transformer.py`.
* `Encoder`: Encoder module implemented in `encoder.py`.
* `Decoder`: Decoder module implemented in `decoder.py`.
* `TrajectoryPredictor`: Trajectory predictor implemented in `predictor.py`.
* `SparseAttentionModel`: Sparse attention model implemented in `model_sparse_attention.py`.
* `StaticFilteringModel`: Static filtering model implemented in `model_static_filtering.py`.

### Datasets
* `TrajectoryDataset`: Trajectory dataset implemented in `trajectory_dataset.py`.
* `WaymoDataset`: Simulated Waymo dataset implemented in `waymo_dataset.py`.

### Utilities
* Visualization utilities implemented in `visualization.py`.
* Layer utilities implemented in `layers.py`.
