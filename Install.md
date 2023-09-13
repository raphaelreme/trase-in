# 0. Download the code

The recommanded way is to use git and cloning the official repository:

```bash
$ git clone https://github.com/raphaelreme/schya.git
```

The code should be organized as

```
DIR/schya/*.py
    LICENSE
    README.md
    main.ipynb
    requirements.txt
    ...
```

If you are using a terminal, please be inside the DIR directory (cd DIR)


# 1. Install Conda (If not already installed)

Follow the online [guidelines](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

# 2. [Optional] Install Mamba (If not already installed)

The installation guidlines have recently changed. You can follow them. Another way is to install it with conda with the following:

```bash
$ conda install -n base -c conda-forge mamba
```

# 3. Create the right python environment

We require python 3.10 so that both byotrack and Caiman can work.

```bash
$ # With mamba
$ mamba create -n utopia python==3.10
$ # Else
$ conda create -n utopia python==3.10
```

# 4. Jump in the environement and install required packaqes

The installation requires [CaImAn](https://github.com/flatironinstitute/CaImAn) and [ByoTrack](https://github.com/raphaelreme/byotrack).

We advise following their respective installation guidelines. Nonetheless we provide here an example of default installation:

```bash
$ # With mamba
$ mamba activate utopia
$ mamba install -c conda-forge caiman  # Install caiman
$ pip install -r requirements.txt  # Install byotrack and additional requirements
$ # Else
$ conda activate utopia
$ conda install -c conda-forge caiman  # Install caiman (Maybe slow with conda)
$ pip install -r requirements.txt  # Install byotrack and additional requirements
```

We do not provide here an example to install with GPU support. Please refer to the documentation of [ByoTrack](https://github.com/raphaelreme/byotrack) and its dependencies.

# 5. You are ready to run the main notebook

You can use the interface of anaconda or a terminal to launch a notebook.


With a terminal:

```bash
$ jupyter-notebook
```
