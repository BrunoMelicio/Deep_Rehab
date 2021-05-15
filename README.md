# DeepRehab

This is a Keras implementation of a modified version of [PersonLab](https://arxiv.org/abs/1803.08225) called **DeepRehab** for Multi-Person Pose Estimation.
The model predicts heatmaps and offsets which allow for computation of 23 joints also known as keypoints, 17 for face and body, and 6 for feet. See the paper for more details. Our implementation can run at high speed on the Edge using Edge TPU devices.


### Requirements

* Linux Ubuntu 16.0 or higher
* Python 2.7
* CUDA 8.0 with cudNN 6.0 or higher
* Conda

### Quick Start

* Run 'conda env create -f environment.yml'.
* Run 'conda activate gitfposenet2'
* Download the [model](https://drive.google.com/file/d/1GydiTWBO9njcIRsc7IzOgUPyi_HQ89k_/view?usp=sharing) and put it inside the `/src/models/` folder.
* Run 'python demo.py' to run the demo and visualize the results inside `/src/demo_results/` or run 'python demo_video.py' to demo on a video.

### Result

**Pose**

![pose](https://github.com/BrunoMelicio/FootPoseNet/blob/main/src/demo_results/keypoints_test.png)


### Advanced

If you want to train the model:

* Download the COCO 2017 dataset and store it in `root/datasets`.

  COCO training images: http://images.cocodataset.org/zips/train2017.zip

  COCO validation images: http://images.cocodataset.org/zips/val2017.zip

  COCO Whole-Body training annotations: https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf/view
  
  COCO Whole-Body validation annotations: https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb/view

  Put the training images in `/datasets/coco/images/train2017/` , val images in `/datasets/coco/images/val2017/`, training and validation annotations in `/datasets/coco/annotations/`. Set the location of the dataset in the config file.

* Generate the training file in the correct format by running 'python generate_hdf5.py'.

* If you want to use Resnet101 as the base, first download the imagenet initialization weights from [here](https://drive.google.com/open?id=1ulygah5BTWjhSGGpN20-eYV5NAozdE8Z) and copy it to your `~/.keras/models/` directory.

* Edit the [config.py](config.py) to set options for training, e.g. input resolution, number of GPUs, whether to freeze the batchnorm weights, etc. More advanced options require altering the [train.py](train.py) script. For example, changing the base network can be done by adding an argument to the get_personlab() function.

* Inside /src/, run 'python train.py'.

If you want to experiment with the filtering methods, please use 'python /src/experiments/filtering_experiments.py'

If you want to run the model on an Edge TPU device, follow these steps:
* Export the frozen graph of the deeprehab_101.h5 model using the jupyter notebook freezeGraph.ipynb . It may require creating a new conda environment with Tensorflow 1.9.
* Convert the exported frozen graph to a tflite model using the jupyter notebook convertToTFLITE.ipynb . It requires to create a new conda environment using Tensorflow nightly 3.5.
* Install the edgetpu compiler following: https://coral.ai/docs/edgetpu/compiler/ .
* Run the command 'edgetpu_compiler [options] model...' , using your own options and the name of the tflite model.

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
