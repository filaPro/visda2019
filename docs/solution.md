Follow the instructions below to reproduce our best accuracy on both tracks.
Note that 8 GPUs with 24 Gb each and ~50 Gb RAM are required to fit the model
with needed batch size. The results on smaller hardware configurations will be worth.

#### Multi-source track

Train 3 models with different backbones and loss weights for clipart domain:
```
python runners/mix_match_multi_source.py --source-domains 0,1,2,3 --target-domain 4 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b5 --name ms-clipart-333-b5
python runners/mix_match_multi_source.py --source-domains 0,1,2,3 --target-domain 4 --n-gpus 8 \
    --batch-size 15 --loss-weight 1000. --backbone efficient_net_b5 --name ms-clipart-1000-b5
python runners/mix_match_multi_source.py --source-domains 0,1,2,3 --target-domain 4 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b6 --name ms-clipart-333-b6
```
Predict the results of all models on test part:
```
python runners/predict.py --name ms-clipart-333-b5 --domain 4 --phase test --track 0 \
    --backbone efficient_net_b5
python runners/predict.py --name ms-clipart-1000-b5 --domain 4 --phase test --track 0 \
    --backbone efficient_net_b5
python runners/predict.py --name ms-clipart-333-b6 --domain 4 --phase test --track 0 \
    --backbone efficient_net_b6
```
Ensemble all predictions:
```
python scripts/ensemble.py --in-names ms-clipart-333-b5,ms-clipart-1000-b5,ms-clipart-333-b6 \
    --out-name ms-clipart --weights .3,.3,.4
```
Process same steps for painting domain:
```
python runners/mix_match_multi_source.py --source-domains 0,1,2,3 --target-domain 5 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b5 --name ms-painting-333-b5
python runners/mix_match_multi_source.py --source-domains 0,1,2,3 --target-domain 5 --n-gpus 8 \
    --batch-size 15 --loss-weight 1000. --backbone efficient_net_b5 --name ms-painting-1000-b5
python runners/mix_match_multi_source.py --source-domains 0,1,2,3 --target-domain 5 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b6 --name ms-painting-333-b6

python runners/predict.py --name ms-painting-333-b5 --domain 5 --phase test --track 0 \
    --backbone efficient_net_b5
python runners/predict.py --name ms-painting-1000-b5 --domain 5 --phase test --track 0 \
    --backbone efficient_net_b5
python runners/predict.py --name ms-painting-333-b6 --domain 5 --phase test --track 0 \
    --backbone efficient_net_b6

python scripts/ensemble.py --in-names ms-painting-333-b5,ms-painting-1000-b5,ms-painting-333-b6 \
    --out-name ms-painting --weights .3,.3,.4
```
Make submission from cliaprt and painting predictions:
```
python make_submission.py --clipart-name ms-clipart --painting-name ms-painting --out-name ms --track 0
```
The submission file is now in `/content/logs/ms`.

#### Semi-supervised track
Process same steps as for multi-source track.

```
python runners/mix_match_semi_supervised.py --target-domain 4 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b5 --name ss-clipart-333-b5
python runners/mix_match_semi_supervised.py --target-domain 4 --n-gpus 8 \
    --batch-size 15 --loss-weight 1000. --backbone efficient_net_b5 --name ss-clipart-1000-b5
python runners/mix_match_semi_supervised.py --target-domain 4 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b6 --name ss-clipart-333-b6

python runners/predict.py --name ss-clipart-333-b5 --domain 4 --phase unlabeled --track 1 \
    --backbone efficient_net_b5
python runners/predict.py --name ss-clipart-1000-b5 --domain 4 --phase unlabeled --track 1 \
    --backbone efficient_net_b5
python runners/predict.py --name ss-clipart-333-b6 --domain 4 --phase unlabeled --track 1 \
    --backbone efficient_net_b6

python scripts/ensemble.py --in-names ss-clipart-333-b5,ss-clipart-1000-b5,ss-clipart-333-b6 \
    --out-name ss-clipart --weights .3,.3,.4

python runners/mix_match_semi_supervised.py --target-domain 5 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b5 --name ss-painting-333-b5
python runners/mix_match_semi_supervised.py --target-domain 5 --n-gpus 8 \
    --batch-size 15 --loss-weight 1000. --backbone efficient_net_b5 --name ss-painting-1000-b5
python runners/mix_match_semi_supervised.py --target-domain 5 --n-gpus 8 \
    --batch-size 15 --loss-weight 333. --backbone efficient_net_b6 --name ss-painting-333-b6

python runners/predict.py --name ss-painting-333-b5 --domain 5 --phase test --track 1 \
    --backbone efficient_net_b5
python runners/predict.py --name ss-painting-1000-b5 --domain 5 --phase test --track 1 \
    --backbone efficient_net_b5
python runners/predict.py --name ss-painting-333-b6 --domain 5 --phase test --track 1 \
    --backbone efficient_net_b6

python scripts/ensemble.py --in-names ss-painting-333-b5,ss-painting-1000-b5,ss-painting-333-b6 \
    --out-name ms-painting --weights .3,.3,.4

python make_submission.py --clipart-name ss-clipart --painting-name ss-painting --out-name ss --track 1
```
The submission file is now in `/content/logs/ss`.