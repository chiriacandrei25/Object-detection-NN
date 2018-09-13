# Description
### Object Detection Neural Network

Program that takes an image or live video feed from camera and uses a Convolutional Neural Network to detect and classify objects in sight.

## Installation
### Dependencies
Tensorflow Object Detection API depends on the following libraries:

- Protobuf 3.0.0
- Python-tk
- Pillow 1.0
- lxml
- tf Slim (which is included in the "tensorflow/models/research/" checkout)
- Jupyter notebook
- Matplotlib
- Tensorflow (>=1.9.0)
- Cython
- contextlib2
- cocoapi

For detailed steps to install Tensorflow, follow the [TensorFlow Installation Instructions](https://www.tensorflow.org/install/). A typical user can install Tensorflow using one of the following commands:

```
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
pip install --user pillow
```

### Coco API Installation
Download the [Coco API](https://github.com/cocodataset/cocoapi) and copy the pycocotools subfolder to the tensorflow/models/research directory if you are interested in using COCO evaluation metrics. The default metrics are based on those used in Pascal VOC evaluation. To use the COCO object detection metrics add ``metrics_set``: ``coco_detection_metric`` to the ``eval_config`` message in the config file. To use the COCO instance segmentation metrics add ``metrics_set``: ``coco_mask_metrics`` to the ``eval_config`` message in the config file.

### Protobuf Compilation
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the ``tensorflow/models/research/`` directory:

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

### Importing the models
You can download the models directory along with a trained Neural Network from [models](https://github.com/tensorflow/models/tree/master/research/object_detection)

### Testing the Installation
You can test that you have correctly installed the Tensorflow Object Detection API by running the following command:

```
python object_detection/builders/model_builder_test.py
```
### Running for images
In the directory containing the models and the source code, create a folder named "test_images" and upload the images for processing.

![](https://github.com/chiriacandrei25/Object-detection-NN/blob/master/screenshot.jpg)

### Running for camera
In case of multiple cameras, choose the one you want to stream from, by setting the index in the line ```capture = cv2.VideoCapture(0/1/2/...)```


### Running Tensorboard
Progress for training and eval jobs can be inspected using Tensorboard. If using the recommended directory structure, Tensorboard can be run using the following command:
```
tensorboard --logdir=${MODEL_DIR}
```
where ```${MODEL_DIR}``` points to the directory that contains the train and eval directories. Please note it may take Tensorboard a couple minutes to populate with data.



