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
