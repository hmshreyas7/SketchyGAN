SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis
=====================================

Code for ["SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis"](https://arxiv.org/abs/1801.02753).


## Prerequisites

- Python 3, NumPy, SciPy, OpenCV 3
- TensorFlow (GPU version)
- A recent NVIDIA GPU (we used Tesla K80)


## Preparation

- In `./src_single/input_pipeline.py`, lines 12-13 need to be modified to point to the correct TFRecords.
- TFRecords are generated using `./data_processing/sketchy_to_tfrecord.py`. Modify lines 37-39 as necessary.
- ["Inception-V4 model"](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) must be downloaded, unzipped, and the checkpoint should be put into `./inception_v4_model`.


## Dataset

**Note**: For Flickr API augmentation, the scripts in `./data_processing` can be used to crawl additional images for training.

- The Sketchy dataset can be found [here](http://sketchy.eye.gatech.edu/). `Sketches and Photos` and `Annotation and Info` are the 2 required links.
- Please contact the author of the [main repository](https://github.com/wchen342/SketchyGAN) if you need the Flickr images used in the original experiments.


## Configurations

Run the following command to initiate the training process from the root directory:

```
python3 main_single.py
```

For more information about commandline options, see `./main_single.py` (lines 148-161).

To analyze the effect of the residual connection, modify equations on lines 275, 342, 521, and 589 in `./src_single/mru.py` to look like this:

```
ht_new = h_new * zg
```

For understanding the impact of the masking mechanism, modify the first argument of the concatenation functions on lines 257, 324, 504, and 572 in the same file to this:

```
[ht, inp]
```

And the equations on lines 275, 342, 521, and 589 to this:

```
ht_new = ht + h_new
```

For the analysis of data augmentation, simply point the Flickr directory to the Sketchy folder instead (wherever necessary).


## Model

- The model will be saved periodically (default: after every 5000 iterations). If you wish to resume, just use the commandline option `resume_from`.
- If you wish to test the model, change `mode` from `train` to `test` and fill in `resume_from`.


## Citation

If you use the author's work for your research, please cite their paper
```
@InProceedings{Chen_2018_CVPR,
author = {Chen, Wengling and Hays, James},
title = {SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```


## Common issues and useful resources

**Issues**:

1. [Segmentation fault caused by importing OpenCV](https://github.com/opencv/opencv/issues/9568)
2. [Training process gets killed](https://github.com/tensorflow/models/issues/3497)
3. [Error importing object detection modules](https://github.com/tensorflow/models/issues/1595)
4. [No module named nets](https://github.com/tensorflow/models/issues/1842)
5. [Coco mask](https://github.com/cocodataset/cocoapi/issues/59)
6. [CUDA dependencies installation](https://github.com/tensorflow/tensorflow/issues/26987)
7. [nvidia-smi](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch)


**Resources**:

1. [Installing pip on Ubuntu 18.04](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
2. [TensorFlow GPU](https://stackoverflow.com/questions/51306862/how-to-use-tensorflow-gpu)
3. [Downloading a file from a website via terminal](https://askubuntu.com/questions/207265/how-to-download-a-file-from-a-website-via-terminal)
4. [Downloading large files from Google Drive](https://pypi.org/project/gdown/)
5. [Extracting from .tar.gz file](https://askubuntu.com/questions/25347/what-command-do-i-need-to-unzip-extract-a-tar-gz-file)
6. [Extracting from .7z file](https://askubuntu.com/questions/219392/how-can-i-uncompress-a-7z-file/219395#219395)
7. [Zipping a folder](https://unix.stackexchange.com/questions/93139/can-i-zip-an-entire-folder-using-gzip)
8. [Counting files in folder and subfolders](https://askubuntu.com/questions/34099/find-number-of-files-in-folder-and-sub-folders)
9. [Deleting n files](https://stackoverflow.com/questions/4817313/bash-script-to-delete-all-but-n-files-when-sorted-alphabetically)
10. [Getting the inception scores](https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter)