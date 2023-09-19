# TraSE-IN


Code of TraSE-IN paper: Automatic monitoring of whole-body neural activity in behaving Hydra.

The tracking code is published as a re-usable python library: [ByoTrack](https://github.com/raphaelreme/byotrack)

![pipeline](pipeline.jpeg)


## Install

The installation requires [CaImAn](https://github.com/flatironinstitute/CaImAn) and [ByoTrack](https://github.com/raphaelreme/byotrack). These two requirements currently impose to have python 3.10.

We advise following their respective installation guidelines. Nonetheless we provide here an example of installation using mamba command (can be replaced by conda but it will be slower)

```bash
$ mamba create -n trasein -c conda-forge python==3.10 caiman
$ mamba activate trasein
$ pip install -r requirements.txt
```

A more complete guide is provided in the `Install.md` file.


## Getting started

The `main.ipynb` notebook runs the whole pipeline and provides visualization at different timestep.

## Model & Data

We have trained our own [StarDist](https://github.com/stardist/stardist) model in order to solve the neurons detection on each frame.
It can be downloaded from https://partage.imt.fr/index.php/s/npwHJHZebxqGMPi. We also provide a downloading script `download_model.sh`

To run detection with this stardist model in the `main.ipynb` notebook, you can simply change `model_path` value with the path to the `stardist` folder.

Example data will soon be available.
