# Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data

[Konstantin A. Maslov](https://people.utwente.nl/k.a.maslov), [Claudio Persello](https://people.utwente.nl/c.persello), [Thomas Schellenberger](https://www.mn.uio.no/geo/english/people/aca/geohyd/thosche/), [Alfred Stein](https://people.utwente.nl/a.stein)

[['Paper']()] [['Datasets'](#datasets)] [['BibTeX'](#citing)] 

<br/>


The repository is in progress.

TODO: Repository/paper description.


## Datasets

The dataset associated with the paper will be uploaded to [the DANS repository](https://dans.knaw.nl/en/) shortly. 

TODO: Share the dataset link, make a subsample (10%?) of the dataset for the demonstration purposes (and share via gdrive?). 


## Installation 

TODO: Required hardware (technically, any might work, but nice GPU and RAM are the only reasonable options computationally), Anaconda/Miniconda, instructions with .yml files, instructions 'from scratch'.

```
conda create -n ... -c conda-forge
```


## Getting started

TODO: Add instructions to the subsections.

### Training a model

```
python train.py ...
```

### Predicting on the test subset

```
python predict.py ...
```

### Evaluating on the test subset

```
python evaluate.py ...
```

### Running on custom/standalone data

```
python compile_features.py ...
python deploy.py ...
```

### Confidence calibration

TODO: Do via a jupyter notebook (refactor and upload).

### Bias optimisation

TODO: Do via a jupyter notebook (refactor and upload).

After that, simply run

```
python deploy.py ... -bias <BIAS VECTOR> ... 
```


## Pretrained models

TODO: Share the weights of the pretrained models (in gdrive if github restrictions do not allow).


## License

This software is licensed under the [GNU General Public License v2](LICENSE).


## Citing

If you use our datasets or models in your research, please use the following BibTeX entry.

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
