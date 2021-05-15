# Foot-PoseNet

This is a Keras implementation of a modified version of [PersonLab](https://arxiv.org/abs/1803.08225) called **Foot-PoseNet** for Multi-Person Pose Estimation.
The model predicts heatmaps and offsets which allow for computation of 23 joints also known as keypoints, 17 for face and body, and 6 for feet. See the paper for more details.


### Requirements

* Linux Ubuntu 16.0 or higher
* Python 2.7
* CUDA 8.0 with cudNN 6.0 or higher
* Conda

### Install

* Run 'conda env create -f environment.yml'.
* Run 'conda activate footposenet'

### Demo

* Download the [model](https://drive.google.com/file/d/1viDeWyRVNwAV6uEw4OcRYrssiSIqSR_a/view?usp=sharing) and put it inside the `root/src/models` folder.
* Run 'python demo.py' to run the demo and visualize the results inside `root/src/demo_results`.

### Result

**Pose**

![pose](https://github.com/BrunoMelicio/FootPoseNet/blob/main/src/demo_results/keypoints_test.png)


### Training

* Download the COCO 2017 dataset and store it in `root/datasets`.

  http://images.cocodataset.org/zips/train2017.zip

  http://images.cocodataset.org/zips/val2017.zip

  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

  training images in `root/datasets/coco2017/images/train2017/` , val images in `root/datasets/coco2017/images/val2017/`, training annotations in `root/datasets/coco2017/annotations/`. Set the location of the dataset in the config file.

* Generate the training file in the correct format by running 'python generate_hdf5.py'.

* If you want to use Resnet101 as the base, first download the imagenet initialization weights from [here](https://drive.google.com/open?id=1ulygah5BTWjhSGGpN20-eYV5NAozdE8Z) and copy it to your `~/.keras/models/` directory.

* Edit the [config.py](config.py) to set options for training, e.g. input resolution, number of GPUs, whether to freeze the batchnorm weights, etc. More advanced options require altering the [train.py](train.py) script. For example, changing the base network can be done by adding an argument to the get_personlab() function.

* Inside root/src, run 'python train.py'.

## Technical Debts
Several parts of this codebase are borrowed from [PersonLab Keras](https://github.com/octiapp/KerasPersonLab)

### Citation

```
@inproceedings{papandreou2018personlab,
  title={PersonLab: Person pose estimation and instance segmentation with a bottom-up, part-based, geometric embedding model},
  author={Papandreou, George and Zhu, Tyler and Chen, Liang-Chieh and Gidaris, Spyros and Tompson, Jonathan and Murphy, Kevin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={269--286},
  year={2018}
}
```
