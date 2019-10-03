This codebase contains the implementation of our (with [@denemmy](https://github.com/denemmy))
solution for [VisDA 2019](http://ai.bu.edu/visda-2019) challenge. 
Our team got **2nd** place on final leaderboard of 
[multi-source](https://competitions.codalab.org/competitions/20256#results) track (with accuracy: .716),
and **3rd** place of [semi-supervised](https://competitions.codalab.org/competitions/20257#results) (with accuracy: .713).
This solution heavily borrows ideas from 
MixMatch ([arxiv](https://arxiv.org/abs/1905.02249), [github](https://github.com/google-research/mixmatch)) 
and EfficientNet ([arxiv](https://arxiv.org/abs/1905.11946), [github](https://github.com/qubvel/efficientnet)).

#### Installation

Just clone this repo and install `requirements.txt` throw `pip`.
The code was tested on `ubuntu 16.04` with `python 3.6`, `cuda 10.0`, `cudnn 7.5`.
You may also need `wget` and `unzip` packages to download data.

#### Data preparation 

Download and convert images to `.tfrecords`:
```
python scripts/download.py
python scripts/convert_to_tfrecords.py
```
The resulting structure of data directory is shown in [docs/structure.md](docs/structure.md).

#### Training example
```
python runners/source_semi_supervised.py
```
The growth of accuracy on sketch domain will be displayed at `stdout` and in log file.
The arguments of all scripts are listed in [docs/arguments.md](docs/arguments.md).

#### Achieving leaderboard accuracy

Follow the instructions in [docs/solution.md](docs/solution.md).

