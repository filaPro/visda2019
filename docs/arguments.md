#### Runners

* `--in-path` - path to data structured as in [docs/structure.md](docs/structure.md)
* `--out-path` - path to save logs and checkpoints
* `--name` - experiment name
* `--time` - set True to add date and time to experiment name
* `--n-gpus` - number of visible GPUs
* `--batch-size` - batch size
* `--n_epochs` - number of training epochs; one epoch is 1000 batches
* `--backbone` - backbone architecture (`efficient_net_b*` with `*` in `{0, 1, 2, 3, 4, 5, 6, 7}`,
  `mobile_net_v2` or `vgg19`)
* `--source-domain` - index of source domain; domains are enumerated as 
  `('real', 'infograph', 'quickdraw', 'sketch', 'clipart', 'painting')`
* `--source-domains` - comma-separated list of source domain indexes
* `--target-domain` - index of target domain
* `--phase` - name of data split; `train` or `test` for multi-source track; 
  `labeled` or `unlabeled` for semi-supervised; `all` can be used for both tracks
  
#### MixMatch runners

* `--loss-weight` - target loss weight
* `--temperature` - distribution sharpening parameter
* `--alpha` - beta distribution parameter

#### Scripts

`convert_to_tfrecords.py`
* `--path` - path to data
* `--size` - number of images per `.tfrecord` file

`download.py`
* `--path` - path to data

`ensemble.py`
* `--in-path` - path to data
* `--out-path` - path to load and save predictions
* `--in-names` - comma-separated names of experiments
* `--out-name` - current experiment name
* `--weights` - comma-separated weights for probabilities

`make_submission.py`
* `--in-path` - path to data
* `--out-path` - path to load and save predictions
* `--clipart-name` - clipart experiment name
* `--painting-name` - painting experiment name
* `--out-name` - current experiment name
* `--track` - `0`: multi-source, `1`: semi-supervised