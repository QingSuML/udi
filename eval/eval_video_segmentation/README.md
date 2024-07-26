### Video object segmentation on DAVIS 2017 
This evaluation follows the settings used in DINO. Please ensure that your environment is configured to match DINO's setup.

**Step 1: Prepare DAVIS 2017 data**  
```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
```

**Step 2: Video object segmentation**  

```
python ./eval_video_segmentation/eval_video_segmentation.py \
    --data_path $DATASET_ROOT \
    --output_dir $OUTPUT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_base \
    --topk 5 \
    --size_mask_neighborhood 12
```

**Step 3: Evaluate the obtained segmentation** 

```
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
python $HOME/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /path/to/saving_dir --davis_path $HOME/davis-2017/DAVIS/
```



## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :blue_book::
```
placeholder
```