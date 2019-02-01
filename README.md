# NumpyYoloProcess


This function takes in the outputs of YOLOv3 (and some parameters), interprets them using Numpy, and returns the detected boxes, box scores, and classes. 

Based heavily on the output-processing code found in [xiaochus's Keras implementation of YOLOv3](https://github.com/xiaochus/YOLOv3), which uses Tensorflow-based code for building a graph for interpreting YOLO outputs. Most of the code in yolo_process.py is just is just a conversion that replaces TF functions with the respective Numpy functions. [Adrian Rosebrock's algorithm](https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/) with a modification for sorting by scores is used for non-max suppression.

##### Parameters

- **yolo_outputs** : a list containing the outputs of YOLOv3
- **anchors**: a parameter specifying which anchors to use (default: 'full')
  - if set to 'full', will use default YOLOv3 anchors 
  - if set to 'tiny', will use default tiny-yolov3 anchors
  - otherwise, it'll expect a numpy array of anchors to use
- **num_classes**: the number of classes (default: 80)
- **image_shape**: a tuple specifying the input image shape, height-first (default: (720,1280))
- **max_boxes**: maximum number of boxes to return (default: 20)
- **score_threshold**: score threshold to use (default: 0.6)
- **iou_threshold**: box overlap theeshold to use (default: 0.5)

##### Returns

- **boxes_**: a list of boxes (top-left-bottom-right lists)
- **scores_**: a list containing the boxes' scores
- **classes_**: a list containing the class predictions for the boxes
