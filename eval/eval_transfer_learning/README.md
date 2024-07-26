### Transfer learning on CIFAR10, CIFAR100, INAT18, INAT19, Flowers and Cars
We follow the setting in iBOT and use the default fine-tuning recipe (i.e., w/o layerwise decay, a smaller learing rate, and a longer training scheduler) proposed in DEiT.

#### CIFAR10
```
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=96
python -m torch.distributed.launch --nproc_per_node=8 ./eval_transfer_learning/eval_transfer_learning.py --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --batch-size 96 \
    --lr 5e-6 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set CIFAR100 
```
Other datasets follow most of the configuration, except that we set `--lr 7.5e-6` for CIFAR100, `--lr 3.0e-5 --epochs 360` for INAT18, `--lr 5.0e-5 --epochs 360` for INAT19, `--lr 7.5e-5` for Flower and Cars.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :blue_book::
```
placeholder
```