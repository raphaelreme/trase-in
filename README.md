# Utopia
Preliminary code for Utopia paper


## Install

The installation requires [CaImAn](https://github.com/flatironinstitute/CaImAn) and [ByoTrack](https://github.com/raphaelreme/byotrack). These two requirements impose to have python 3.10.

We advise following their respective installation guidelines. Nonetheless we provide here an example of installation using mamba command (can be replaced by conda but it will be slower)

```bash
$ mamba create -n utopia -c conda-forge python==3.10 caiman
$ mamba activate utopia
$ pip install -r requirements.txt
```
