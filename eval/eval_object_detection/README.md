## Evaluating object detection and instance segmentation on the COCO dataset

**Prerequisite: mmcv 1.7.0**

### Step 1. Prepare COCO dataset  

The dataset can be downloaded at [`https://cocodataset.org/#download`](https://cocodataset.org/#download)  

### Step 2. Install mmdetection  

```
git clone -b v2.26.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection/
pip install -r requirements/build.txt
pip install -v -e .
```

### Step 3. Include the files in `backbone`, `configs`, `tools` into the mmdetection library

### Step 4. Fine-tune on the COCO dataset  

```
tools/dist_train.sh configs/udi/mask_rcnn_vit_small_12_p16_1x_coco.py [number of gpu]\  
--work-dir /path/to/saving_dir\
--seed 0 --deterministic\
--options model.pretrained=/path/to/model_dir\  
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :blue_book::
```
placeholder
```