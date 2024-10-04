# DSen2
Deep Sentinel-2 and modifications

[Super-Resolution of Sentinel-2 Images: Learning a Globally Applicable Deep Neural Network](https://arxiv.org/abs/1803.04271)

Original author: Charis Lanaras, charis.lanaras@alumni.ethz.ch

## Requirements

- Tensorflow (no more tensorflow-gpu after 2.14)
- Keras
- Numpy
- Scikit-image
- argparse
- Rasterio
- Matplotlib
- Cartopy
- Sympy
- Osgeo
- Geopandas
- SentinelSat
- joblib
- Pillow
- Pandas
- YAML/PyYAML

## Training

Training is handled by reading and processing configuration YAML files in the `config` directory. During training time, the config files are copied to the run directory of the simulation. YAML files look like this:

```
---
default:
  run_id: 'firewx-dsen2_mae_and_dssim'
  restart: false
  model: 'dsen2'
training:
  restart:
    path: /home/dryglicki/code/DSen2/training/firewx-unet_mae/best_checkpoints/best-min_val_loss-epoch_0200.hdf5
    epoch: 200
  directory:
    training_hires:    /home/dryglicki/data/sentinel-2/paired/training/red
    training_lowres:   /home/dryglicki/data/sentinel-2/paired/training/nirLow
    truth_train:       /home/dryglicki/data/sentinel-2/paired/training/nirHigh
    validation_hires:  /home/dryglicki/data/sentinel-2/paired/validation/red
    validation_lowres: /home/dryglicki/data/sentinel-2/paired/validation/nirLow
    truth_valid:       /home/dryglicki/data/sentinel-2/paired/validation/nirHigh
  dataset:
    batch_size: 48
    shuffle_size: 64
    augment: true
...
```

Explore the `config` directory for further examples.

## Using a Trained Network

In the `training` directory, weights and serialized models can be found in the directories of each simulation that is uploaded to github.

## Downloading/preprocessing Sentinel-2 data
1. Create a JSON file that creates a bounding box for a Sentinel-2 API search. [This link](https://geojson.io) can be used build one.
2. Use `auxiliary/sentinel-2/download/save_to_GeoJSON.py` to read the JSONs created by the previous step. This step checks for availability based on cloud cover, geographic location,
and time of year. The output of this would be GeoJSON files.
3. Use `auxiliary/sentinel-2/download/download_in_chunks.py` to read the GeoJSON files and download the data.
4. Use `auxiliary/sentinel-2/image_processing/create_paired_training_set_red-nir.py` to create the training/validation triplets: VIS High-Res, NIR Low-Res, and NIR High-Res.

## Model Training

Training is handled by `training/quick_training.py`. It will utilize a user's config file from `config/`.
```
usage: quick_training.py [-h] [--noclobber] yaml_file

Training a pan-sharpening neural network.

positional arguments:
  yaml_file    YAML file with training guidelines.

options:
  -h, --help   show this help message and exit
  --noclobber  Do not clobber existing directories.
```
