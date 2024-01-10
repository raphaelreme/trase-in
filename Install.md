# 0. Download the code

The recommended way is to use git and cloning the official repository:

```bash
$ git clone https://github.com/raphaelreme/trase-in.git
```

The code should be organized as

```
DIR/trasein/*.py
    LICENSE
    README.md
    main.ipynb
    requirements.txt
    ...
```

If you are using a terminal, please be inside the DIR directory (cd DIR)


# 1. Install Conda (If not already installed)

Conda is a package manager, usually used to manage python environment. It will help you setup the correct environement for TraseIn.

Follow the online [guidelines](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

# 2. [Optional] Install Mamba (If not already installed)

Mamba is a much faster implementation of conda API. It is useful to solve quickly packages dependencies and installations.
You can do without mamba but it is gonna take more time.

We advise following the online installation guidelines. Nonetheless here is a simple way to install it through conda:

```bash
$ conda install -n base -c conda-forge mamba
```

# 3. Create the right python environment

We require python 3.10 so that both ByoTrack and Caiman can work.

```bash
$ # With mamba
$ mamba create -n trasein python==3.10
$ # Else
$ conda create -n trasein python==3.10
```

Your new python environement is called `trasein`. In a terminal, you can use it with:

```bash
$ # With mamba
$ mamba activate trasein   # You may have to still use 'conda activate' instead of 'mamba activate'
$ # Else
$ conda activate trasein
```

Once activated, you can install additional python libraries in the env, launch a python shell or script with the env. 

# 4. Install required packaqes

The installation requires [CaImAn](https://github.com/flatironinstitute/CaImAn) and [ByoTrack](https://github.com/raphaelreme/byotrack).

We advise following their respective installation guidelines. Nonetheless we provide here an example of default installation:

```bash
$ # With mamba
$ mamba activate trasein  # Activate the env if not done (you may have to still use 'conda activate' instead of 'mamba activate')
$ mamba install -c conda-forge caiman  # Install caiman
$ pip install -r requirements.txt  # Install byotrack and additional requirements
$ # Else
$ conda activate trasein  # Activate the env if not done
$ conda install -c conda-forge caiman  # Install caiman (Slow with conda)
$ pip install -r requirements.txt  # Install byotrack and additional requirements
```

We do not provide an example to install with GPU support. Running on GPU allows you to run faster the StarDist detection that we use in TraseIn. Please refer to the [documentation of ByoTrack](https://byotrack.readthedocs.io/en/latest/install.html) and its dependencies (specially StarDist and Tensorflow).

# 5. Install Icy
In TraseIn, the tracking is done in three steps: Detections on each frame, linking detections into tracklets, tracklet-stitching to bridge over missed detections.

Detections is done in python with StarDist. The tracklet stitching part is implemented directly in python in ByoTrack. But the linking of detections through time is based on
EMHT algorithm, implemented in java in Icy software. Thus, to run the tracking pipeline, TraseIn requires Icy as an external dependency. (The python code calls Icy and execute EMHT
algorithm in java).

Follow the [installation guide of Icy](https://icy.bioimageanalysis.org/tutorial/installation-instructions-for-icy-software/). And check that you are able to run the application.

Locate the Icy installation folder. For Windows or Linux, you have chosen this location. For MacOs, it may be automatically installed in the Applications folder. Inside of this installation folder, you should find some executable for Icy and the main jar file: `icy.jar`. The path to this jar file should be given to ByoTrack in the tracking pipeline. (See `Run the main notebook`)

# 6. Download StarDist model

We have trained our own [StarDist](https://github.com/stardist/stardist) model in order to detect neurons on each frame.
It can be downloaded from https://partage.imt.fr/index.php/s/npwHJHZebxqGMPi. We also provide a downloading script `download_model.sh`

In a terminal, you can simply do:
```bash
$ bash download_model.sh  # Download and unzip the StarDist trained model
```

Or manually download the zip file and unzip it. The path to the stardist folder should be given to ByoTrack for the detection process. (See `Run the main notebook`)

# 7. Download Example Data

You can use your own videos, but for reproduction purposes we have uploaded our own data. It will be soon available.

Download and unzip the data. The path to a GCaMP video and TdTomato video must be provided as input of the whole pipeline. (See `Run the main notebook`)

# 8. Run the main notebook

You can use the interface of anaconda or a terminal to launch a notebook.

With a terminal (First activate the `trasein` env):

```bash
$ jupyter-notebook
```

You can run the `main.ipynb` notebook cell by cell.

The notebook requires 4 different paths to be set inside the cells:
- `gcamp_path`: path to the GCaMP video (for instance "path/to/data/doi_10_5061_dryad_h9w0vt4q3__v20230925/G7_contrxn-1.avi")
- `tdtomato_path`: path to the TdTomato video (for instance "path/to/data/doi_10_5061_dryad_h9w0vt4q3__v20230925/tdt_contrxn-1.avi")
- `model_path`: path to the trained stardist model (for instance "./stardist", if downloaded and unzipped in the current folder)
- `icy_path`: path to the main Icy jar (for instance "path/to/Icy/icy.jar")

# TroubleShooting

## Known errors
### Running Icy for the first time fails
The first time you run EMHT (Icy) with our python wrapper in a long time, it may fail because Icy updates its plugins and exits... 
The issue is not fixed yet, but you can simply run again the code/cell and it will now work.
