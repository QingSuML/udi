## Evaluating semantic segmentation on the ADE20K dataset

**Prerequisite: mmcv 1.7.0**

### Step 1. Prepare ADE20K dataset

The dataset can be downloaded at 
`http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl`

or following instruction of `https://github.com/CSAILVision/ADE20K`

### Step 2. Install mmdetection  

```
git clone -b v0.30.0 https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation/
pip install -v -e .
```

### Step 3. Include the files in `backbone`, `configs`, `tools` into the mmdetection library

### Step 4. Convert your model

```
python tools/model_converters/vit2mmseg.py /path/to/model_dir /path/to/saving_dir
```

### Step 5. Fine-tune on the ADE20K dataset

```
tools/dist_train.sh configs/udi/semfpn_vit-s16_512x512_40k_ade20k.py [number of gpu]\
--work-dir /path/to/saving_dir\
--seed 0 --deterministic\
--options model.pretrained=/path/to/model_dir
```
The optimization hyperarameters are adopted from <a href=https://github.com/facebookresearch/xcit>XCiT</a>.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :blue_book::
```
placeholder
```