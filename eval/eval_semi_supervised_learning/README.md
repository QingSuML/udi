### Semi-supervised classification on ImageNet-1K

We use the data split defined in SimCLRv2, see [here](https://github.com/google-research/simclr/tree/master/imagenet_subsets).

#### fine-tuning

```
torchrun --standalone --nproc_per_node=4 ./eval_semi_supervised_learning/eval_semi_supervised_learning.py --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --lr 5.0e-6 \
    --target_list_path $path_to_the_split.txt
```

#### logistic regression

```
torchrun --standalone --nproc_per_node=4 ./eval_semi_supervised_learning/eval_logistic_regression.py --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --target_list_path $path_to_the_split.txt
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :blue_book:
```
placeholder
```