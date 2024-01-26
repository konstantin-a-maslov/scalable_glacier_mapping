# Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data

[Konstantin A. Maslov](https://people.utwente.nl/k.a.maslov), [Claudio Persello](https://people.utwente.nl/c.persello), [Thomas Schellenberger](https://www.mn.uio.no/geo/english/people/aca/geohyd/thosche/), [Alfred Stein](https://people.utwente.nl/a.stein)

[['Paper']()] [['Datasets'](#datasets)] [['BibTeX'](#citing)] 

<br/>


**The repository is in progress**.

![GlaViTU](assets/glavitu.png)

This GitHub repository is dedicated to the suite of tools and techniques developed in our study "Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data." 
It provides access to our developed convolutional-transformer deep learning model, Glacier-VisionTransformer-U-Net (GlaViTU), and the multimodal dataset, including optical and SAR satellite data, used for our analyses. 
The repository is structured to facilitate a deeper understanding of our methodology in global-scale glacier mapping, offering insights into our model training and generalization strategies. 
By sharing these resources, we aim to support further research and collaboration in the field of environmental monitoring and glacier mapping using artificial intelligence.


## Datasets

The dataset associated with the paper will be uploaded to [the DANS repository](https://dans.knaw.nl/en/) shortly. 

TODO: Share the dataset link, make a subsample (10%?) of the dataset for the demonstration purposes (and share via gdrive?). 

- Dataset
- Demo dataset
- Standalone dataset


## Installation 

### Required hardware

We recommend using a machine with at least 24 GB GPU RAM and 64 RAM. 
Technically, any modern computer is suitable to run the provided code. 
However, no GPU or not enough RAM can make the computational time unreasonably long (up to months and even years). 
In case if your RAM is not enough to perform inference on your own data with our pretrained models, consider splitting it into smaller subareas. 
If your machine lacks GPU RAM, you may consider reducing the batch size by modifying `batch_size = ...` in `configs/data.py` accordingly. 
Please note that altering the batch size can potentially change the expected performance of the models if you train them from scratch.


### Instructions

We recommend using the [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) Python distributions. 
After installing one of them, one can use the `conda` package manager to install the required libraries in a new environment called `massive-tf` and activate it by running

```
conda create -n massive-tf "tensorflow>=2.7" h5py scikit-learn rioxarray geopandas jupyterlab tqdm -c conda-forge
conda activate massive-tf
```

We tested this configuration on Ubuntu 20.04 and Ubuntu 22.04 (see `env_ub2004.yml` and `env_ub2204.yml` for tested dependencies). 
We also expect it to work on any modern Linux distribution or Windows, given properly configured NVIDIA GPU drivers.


## Getting started

TODO: Add instructions to the subsections.

### Adjusting configs

### Training/finetuning a model

```
(massive-tf) python train.py ...
```

### Predicting on the test subset

```
(massive-tf) python predict.py ...
```

### Evaluating on the test subset

```
(massive-tf) python evaluate.py ...
```

### Running on custom/standalone data

```
(massive-tf) python compile_features.py ...
(massive-tf) python deploy.py ...
```

### Confidence calibration

TODO: Do via a jupyter notebook (refactor and upload).

### Bias optimisation

TODO: Do via a jupyter notebook (refactor and upload).

After that, simply run

```
(massive-tf) python deploy.py ... -bias <BIAS VECTOR> ... 
```


## Pretrained models

TODO: Share the weights of the pretrained models (in gdrive if github restrictions do not allow).


## License

This software is licensed under the [GNU General Public License v2](LICENSE).


## Citing

To cite the paper/repository, please use the following bib entry. 

```
@article{towardsglobalglaciermapping2024,
    title={Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data},
    author={Maslov, Konstantin A. and Persello, Claudio and Schellenberger, Thomas and Stein, Alfred},
    journal={},
    year={2024},
    volume={},
    number={},
    pages={},
    doi={}
}
```


<br/>

> If you notice any inaccuracies, mistakes or errors, feel free to submit a pull request or kindly email the authors.
